import asyncio
import datetime
import hashlib
import json
import logging
import numpy as np
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from typing import Iterable, Union

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import time
from copy import deepcopy

from ezmt.organism import Organism, dna2str
from ezmt.plotting import plot_generation_scores
from ezmt.common_funcs import resolve_log_states
from ezmt.config_validation import validate_config
from ezmt.the_pickler import check_state_picklability

# Written at the version level. Its presence is what marks a folder as a run
# rather than a generation, which is how Organism.load resolves "latest".
RUN_SUMMARY_FILE = "run_summary.json"

_log = logging.getLogger("ezmt.tuner")

# A checkpoint folder holds the state file, the DNA prefix it belongs to (for
# reading the tree by hand), and a marker written last. Resuming requires the
# marker, so a checkpoint interrupted part-way through writing is ignored rather
# than loaded half-formed.
CHECKPOINT_STATE_FILE = "state.json"
CHECKPOINT_PREFIX_FILE = "prefix.txt"
CHECKPOINT_MARKER_FILE = "complete"

# Folder names are a digest of the DNA prefix: prefixes run to hundreds of
# characters and contain '(', ',' and '=', so they are neither short enough nor
# legal as path components.
_DIGEST_LENGTH = 16


class ModelTuner:

    def __init__(
            self,
            model_space: list,
            hyperparam_space: dict,
            save_load_funcs: dict = None,
            data: pd.DataFrame = None,
            y_col: str = None,
            generations: int = 1,
            pop_size: int = 20,
            goal: str = 'min',
            directory: str = "organisms",
            sequential: bool = False,
            temp_directory: str = None,
            cleanup_temp: bool = False,
            save_organisms: str = "best",
    ):
        """
        Args:
            sequential: Run organisms one at a time, spilling the DNA prefixes
                they share to disk instead of holding every branch in memory.
                Needed when the branches are too large to coexist.
            temp_directory: Where spilled prefixes and in-progress organisms
                live. Defaults to a hidden folder beside the organisms.
            cleanup_temp: Delete that tree once the run finishes.
            save_organisms: "best" saves only the winner; "all" additionally
                keeps every organism's folder for comparison.
        """
        if save_organisms not in ("best", "all"):
            raise ValueError(
                f'save_organisms must be "best" or "all". Got {save_organisms!r}'
            )
        # Generations can be used for batches of data and not for evolution
        self.model_space = validate_config(model_space, hyperparam_space)
        self.hyperparam_space = hyperparam_space
        self.save_load_funcs = save_load_funcs if save_load_funcs else {}
        self.gpu_semaphore = asyncio.Semaphore(1)  # Used to ensure only 1 process is accessing the GPU at a time
        self.data_fold_generator = generate_stratified_folds(data, y_col)
        # Whether every generation gets the same fold. When data is None the
        # generator yields the same empty fold forever, so a prefix cached in one
        # generation is still valid in the next; when it is not, each generation
        # trains on different rows and cached prefixes must not be shared.
        self.folds_are_static = data is None
        self.generations = generations
        self.population_size = pop_size
        self.population = []
        self.goal = goal
        self.directory = directory
        self.sequential = sequential
        self.temp_directory = temp_directory or f"{directory}/.ezmt_tmp"
        self.cleanup_temp = cleanup_temp
        self.save_organisms = save_organisms
        self.fold_key = "0"
        self.metrics = []
        # One entry per generation, each a list of per-organism scalars. Kept so
        # a caller can persist a whole run without reaching into tuner
        # internals: run() returns it, and it is what run_summary.json holds.
        self.generation_history = []
        self.started_at = None
        self.run_version = None
        self.run_name = None
        self.current_generation = 0

    def populate_init(self, run_name):
        # Generate initial population
        for _ in range(self.population_size):
            dna = choose_dna(self.model_space)
            hyperparams = choose_hyperparams(self.hyperparam_space)
            organism = Organism(run_name, dna, hyperparams, save_load_funcs=self.save_load_funcs, directory=self.directory)
            self.population.append(organism)

    def select_and_reproduce(
            self,
            elitism=1,
            reproduction='asexual',
            gene_mutate_prob=0.05,
            nuc_mutate_prob=0.1,
    ):
        """Build the next generation from this one.

        Args:
            elitism: How many top organisms carry forward unchanged. Forced to
                at least 1: with 0 the run can end on a generation worse than one
                it already had, and the result it returns is not the best it
                found.
            reproduction: 'asexual' or 'sexual'.
            gene_mutate_prob: Per-gene chance of a function mutation.
            nuc_mutate_prob: Per-hyperparameter chance of a value mutation.
        """
        elitism = max(1, elitism)
        # Generate population from previous generation
        # keep top models
        self.population = sorted(self.population, reverse=True)
        new_pop = self.population[0:elitism]
        # Deliberately NOT reset. These organisms already ran, and their score is
        # the measured one. Clearing it made the next generation re-run them, and
        # training is stochastic -- so an elite could come back worse than the
        # score that earned it the slot, and did: a run scored 0.002214 in
        # generation 1 and returned 0.001206 from generation 2, having bred away
        # the better organism it had already found. Keeping saved_result is what
        # the run loop uses to recognise them.

        # Determine who survives and can reproduce
        survivors = natural_selection(self.population)
        # Reproduce the population
        for _ in range(len(self.population) - len(new_pop)):
            if reproduction == 'asexual':
                parent = random.choice(survivors)
                child = parent.reproduce()
            elif reproduction == 'sexual':
                parent1, parent2 = random.sample(survivors, 2)
                child = parent1.mate(parent2)
            else:
                raise ValueError(f'Cannot reproduce with reproduction type: {reproduction}')
            mutate(
                child,
                self.model_space,
                self.hyperparam_space,
                gene_mutate_prob,
                nuc_mutate_prob,
            )
            new_pop.append(child)
        self.population = new_pop

    async def _hold_gpu_semaphore(self, coro):
        """Await ``coro`` while holding the GPU semaphore.

        The semaphore has to be acquired inside the coroutine that does the work.
        Acquiring it around asyncio.create_task only covers scheduling: the task
        is queued and returns immediately, so the semaphore is released before
        the gene runs and serialises nothing.

        Args:
            coro: The gene coroutine to run under the semaphore.

        Returns:
            Whatever ``coro`` returns.
        """
        async with self.gpu_semaphore:
            return await coro

    async def experience_population(self, state, pool, log_states=False):
        # Just as we experience the universe, the universe experiences us
        states_to_log = resolve_log_states(log_states)
        if self.sequential:
            return await self._experience_sequentially(state, states_to_log)
        return await self._experience_concurrently(state, pool, states_to_log)

    # ------------------------------------------------------------------
    # Sequential execution
    # ------------------------------------------------------------------

    async def _experience_sequentially(self, base_state, states_to_log):
        """Run organisms one at a time, resuming from shared prefixes on disk.

        The concurrent path is gene-major: it advances every DNA branch through
        one gene before moving on, so every branch's state is live at once. Here
        one organism runs to completion before the next starts, and the prefixes
        the population shares are written to disk at the point it forks. Peak
        memory is one organism instead of one per branch.

        Args:
            base_state: The starting state for an organism with no cached prefix.
            states_to_log: Gene indices to log state after, or None for all.

        Returns:
            {dna string: saved state}, where each saved state holds file paths
            and scalars rather than the objects themselves.
        """
        checkpoints = find_checkpoint_prefixes(self.population)
        _log.info(
            "Sequential run: %d organism(s), %d shared checkpoint(s)",
            len(self.population), len(checkpoints),
        )
        results = {}
        folders = {}
        total = len(self.population)
        for n, organism in enumerate(self.population, start=1):
            dna = dna2str(organism.dna, organism.parameters)
            if dna in results:
                _log.info("Organism %d/%d: identical genome already run", n, total)
                # Point it at the twin that did run. Leaving it on the folder
                # populate_init handed out is worse than useless: every organism
                # gets the same timestamped name, so it aliases whichever
                # organism is published there at the end of the run.
                organism.folder = folders[dna]
                continue
            # An organism with a saved result already ran, in an earlier
            # generation, and select_and_reproduce kept it rather than resetting
            # it. Re-running it would spend a full training to answer a question
            # already answered, and answer it differently, because training is
            # stochastic. The carried value is the saved dict of file paths, not
            # live state, so this costs no memory.
            #
            # Only valid while the folds are static. With a rolling fold each
            # generation trains on different rows, so a score carried across one
            # is not comparable to its peers and the organism has to run again.
            carried = organism.saved_result if self.folds_are_static else None
            if carried:
                # Keeps the folder it ran in. Reassigning one here would point
                # the summary at a generation that never wrote this organism.
                _log.info("Organism %d/%d: carried forward, not re-run", n, total)
                results[dna] = carried
                folders[dna] = organism.folder
                continue

            # Generation and index are both in the path, so a later generation
            # cannot land on an earlier one's folder.
            organism.folder = self.organism_folder(self.current_generation, n - 1)
            folders[dna] = organism.folder
            state, first_gene = self.restore_from_checkpoint(organism, base_state)
            _log.info(
                "Organism %d/%d: starting at gene %d/%d",
                n, total, first_gene + 1, len(self.model_space),
            )
            for i in range(first_gene, len(self.model_space)):
                state = await organism.run_gene(
                    'train', i, state,
                    log_state=states_to_log is None or i in states_to_log,
                )
                self.save_fork_checkpoint(organism, i, state, checkpoints)
            organism.knowledge = state
            results[dna] = organism.save()
            # Kept so a later generation can carry this organism forward without
            # re-running it. Paths and scalars only.
            organism.saved_result = results[dna]
            # The saved dict names every output as a file, so dropping the live
            # state here is what actually frees the model and its datasets.
            organism.knowledge = {}
        return results

    def checkpoint_folder(self, prefix):
        """Return the folder a DNA prefix's cached state lives in."""
        digest = hashlib.sha1(prefix.encode("utf-8")).hexdigest()[:_DIGEST_LENGTH]
        return f"{self.temp_directory}/fold_{self.fold_key}/{digest}"

    def restore_from_checkpoint(self, organism, base_state):
        """Load the deepest cached prefix this organism can resume from.

        Deliberately re-reads from disk rather than handing out a state another
        organism is already using. Without a process pool nothing pickles the
        state between genes, so two organisms sharing one loaded object would see
        each other's in-place edits.

        Args:
            organism: The organism about to run.
            base_state: The state to start from when nothing is cached.

        Returns:
            (state, first_gene) -- the state to run from and the gene to run next.
        """
        for i in range(len(organism.dna) - 1, -1, -1):
            prefix = dna2str(organism.dna[:i + 1], organism.parameters)
            folder = self.checkpoint_folder(prefix)
            if not os.path.exists(os.path.join(folder, CHECKPOINT_MARKER_FILE)):
                continue
            _log.info("Resuming from cached prefix after gene %d", i + 1)
            state = Organism.load_state(
                folder, CHECKPOINT_STATE_FILE, organism.save_load_funcs
            )
            return state, i + 1
        return dict(base_state), 0

    def save_fork_checkpoint(self, organism, gene_index, state, checkpoints):
        """Spill the state to disk when the population forks after this gene.

        Args:
            organism: The organism that just ran the gene.
            gene_index: Index of the gene that just ran.
            state: The state it produced.
            checkpoints: Prefixes worth caching, from find_checkpoint_prefixes.
        """
        prefix = dna2str(organism.dna[:gene_index + 1], organism.parameters)
        if prefix not in checkpoints:
            return
        folder = self.checkpoint_folder(prefix)
        marker = os.path.join(folder, CHECKPOINT_MARKER_FILE)
        if os.path.exists(marker):
            return  # an earlier organism already cached this prefix
        organism.save_state(folder, CHECKPOINT_STATE_FILE, state)
        with open(os.path.join(folder, CHECKPOINT_PREFIX_FILE), "w") as f:
            f.write(prefix)
        # Written last, so an interrupted save leaves a folder that is skipped
        # rather than one that loads as a truncated state.
        open(marker, "w").close()
        _log.info(
            "Cached prefix after gene %d for %d organism(s)",
            gene_index + 1, checkpoints[prefix],
        )

    # ------------------------------------------------------------------
    # Concurrent execution
    # ------------------------------------------------------------------

    async def _experience_concurrently(self, state, pool, states_to_log):
        # for each decision point, process only unique chains of decisions + args from first decision point to current
        unique_organisms = {'': state}

        for i in range(len(self.model_space)):
            _log.info("Processing gene %d/%d for each organism", i + 1, len(self.model_space))
            prev_unique = unique_organisms
            unique_organisms = dict()

            # Start all processes for this section of DNA. Only process unique decisions based on populations' DNAs
            for organism in sorted(
                    self.population,
                    key=lambda org: (False if org.dna[i]['train'] is None else True) and org.dna[i]['train']['gpu']
            ):
                current_dna = dna2str(organism.dna[:i + 1], organism.parameters)
                _log.debug("Processing organism branch %s", current_dna)
                # check if identical series of decisions up to this stage has already started calculating
                if current_dna in unique_organisms.keys():
                    continue

                prev_dna = dna2str(organism.dna[:i], organism.parameters)
                # state is a dict of that hold all the saved outputs from previous steps for later use
                try:
                    # Must wrap in dict to make a shallow copy
                    # I don't think we need to make a deep copy because we only overwrite keys on the outer layer
                    state = dict(prev_unique[prev_dna])
                except KeyError:
                    raise KeyError(f'No state found for previous dna: {prev_dna}')

                gene_run = organism.run_gene(
                    'train', i, state, pool, log_state=states_to_log is None or i in states_to_log
                )
                if organism.dna[i]['train']['gpu']:
                    gene_run = self._hold_gpu_semaphore(gene_run)
                new_state = asyncio.create_task(gene_run)
                # TODO organisms with current step param run_in_parent_process=True should be sorted to the end of the organisms
                #  so that other async/parallel jobs can start first.
                #  Also it should be assigned False by default for train branch in config validation

                unique_organisms[current_dna] = new_state

            # Wait for all processes for this decision point to complete
            for dna, output in unique_organisms.items():
                unique_organisms[dna] = await output

        # Each organism "remembers" what it has processed
        # knowledge is saved when the organism is saved, and it is loaded later to use during inference
        # TODO Shouldn't this just be done inside organism.run_gene?
        for organism in self.population:
            organism.knowledge = unique_organisms[
                dna2str(organism.dna, organism.parameters)
            ]

        return unique_organisms


    def score_fitness(self, unique_organisms):
        # TODO fix nan
        scores = []
        if self.goal == 'min':
            worst = -np.inf
            worst_dna = None
            best = np.inf
            best_dna = None
            for dna, data in unique_organisms.items():
                scores.append(data['score'])
                if data['score'] < best:
                    best = data['score']
                    best_dna = dna
                if data['score'] > worst:
                    worst = data['score']
                    worst_dna = dna

        else:
            worst = np.inf
            worst_dna = None
            best = -np.inf
            best_dna = None
            for dna, data in unique_organisms.items():
                scores.append(data['score'])
                if data['score'] > best:
                    best = data['score']
                    best_dna = dna
                if data['score'] < worst:
                    worst = data['score']
                    worst_dna = dna

        self.record_generation()
        self.metrics.append({'unique_organisms': len(unique_organisms.keys()),
                             'average': np.mean(scores),
                             'variance': np.var(scores),
                             'best': best,
                             'best_dna': best_dna,
                             'worst': worst,
                             'worst_dna': worst_dna})

        for model in self.population:
            dna = dna2str(model.dna, model.parameters)
            model.score = unique_organisms[dna]['score']
            scores.append(model.score)
            if len(unique_organisms) == 1:
                model.fitness = 1
            else:
                model.fitness = (model.score - worst) / (best - worst)

    async def run(self, run_name, log_states: Union[bool, int, Iterable[int]] = False):
        """Run every generation, then publish a run summary.

        Args:
            run_name: Name this run's organisms are stored under.
            log_states: Gene indices to log state after, True for all.

        Returns:
            (best_organism, run_summary). The organism comes back hydrated
            because callers read its knowledge; the summary carries every
            organism of every generation, so a caller can record the whole run
            without reaching into this object.
        """
        _log.debug("model_space: %s", self.model_space)
        self.run_name = run_name
        self.started_at = int(time.time())
        self.run_version = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with ExitStack() as stack:
            # Nothing runs concurrently in sequential mode, so a pool would only
            # add worker processes and force every sync gene to pickle the whole
            # state across a process boundary. Without one those genes run in a
            # thread and share the state by reference.
            pool = None if self.sequential else stack.enter_context(ProcessPoolExecutor(8))
            for gen in range(self.generations):
                if gen == 0:
                    self.populate_init(run_name)
                else:
                    self.select_and_reproduce()
                _log.info("Starting generation %d/%d", gen + 1, self.generations)
                for organism in self.population:
                    _log.debug("Population member: %s", organism)
                t = time.time()
                # Get next fold of data for next generation
                x_train, x_test, y_train, y_test = next(self.data_fold_generator)
                # Cached prefixes are only interchangeable between generations
                # that train on the same rows.
                self.current_generation = gen + 1
                self.fold_key = "static" if self.folds_are_static else str(gen)
                results = await self.experience_population(
                    {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test},
                    pool,
                    log_states=log_states
                )

                if not self.sequential:
                    # Folders have to exist before score_fitness records the
                    # generation, because the record names where each organism
                    # actually landed.
                    self.persist_generation()
                self.score_fitness(results)
                _log.info("Generation %d runtime: %.1fs | metrics: %s", gen + 1, time.time() - t, self.metrics[-1])
                for model in self.population:
                    _log.debug("Organism DNA: %s", model.dna)

            # Across every generation, not just the last: without elitism a
            # generation can end worse than one before it, and the run should
            # not return something it already beat.
            best_entry = self.find_best_entry()
            best = self.load_best_organism(best_entry)
            summary = self.publish_run_summary(best_entry)
            if self.save_organisms == "best":
                self.prune_losing_organisms(best_entry)
            if self.cleanup_temp:
                shutil.rmtree(self.temp_directory, ignore_errors=True)
            return best, summary

    def load_best_organism(self, best_entry):
        """Return the winning organism, hydrated, from its own folder.

        It is not copied up to the version folder. The summary records where it
        lives and this reads it back, so there is exactly one copy of any
        organism's state.

        Args:
            best_entry: The winning organism's generation_history entry.

        Returns:
            The organism, with knowledge loaded.
        """
        best = max(self.population)
        if best_entry is None:
            return best
        folder = f"{self.run_folder()}/{best_entry['folder']}"
        best.folder = folder
        if os.path.exists(os.path.join(folder, "knowledge.json")):
            best.knowledge = Organism.load_state(
                folder, "knowledge.json", best.save_load_funcs
            )
        best.score = best_entry["score"]
        best.fitness = best_entry["fitness"]
        return best

    def persist_generation(self):
        """Write this generation's organisms to their folders.

        Only for the concurrent path. The sequential loop already saves each
        organism as it finishes, because it has to release the state to keep
        peak memory to one organism.

        Organisms sharing a genome share one result, so they share one folder --
        writing a copy per position would duplicate identical state.
        """
        written = {}
        for i, organism in enumerate(self.population):
            dna = dna2str(organism.dna, organism.parameters)
            if dna in written:
                organism.folder = written[dna]
                continue
            organism.folder = self.organism_folder(self.current_generation, i)
            written[dna] = organism.folder
            organism.save()

    def prune_losing_organisms(self, best_entry):
        """Delete every organism folder except the winner's.

        Organisms now write straight to their final generation/organism folder
        rather than to a temp tree the run throws away, so without this a run
        keeps every organism's datasets and model weights -- about 20x the
        winner alone, and the reason hundreds of GB had to be reclaimed by hand.

        Their scores, genomes and hyperparameters survive in run_summary.json,
        which is what anything downstream actually reads.

        Args:
            best_entry: The winning organism's generation_history entry, or None.
        """
        keep = best_entry["folder"] if best_entry else None
        for generation in sorted(os.listdir(self.run_folder())):
            generation_path = os.path.join(self.run_folder(), generation)
            if not generation.isdigit() or not os.path.isdir(generation_path):
                continue
            for index in sorted(os.listdir(generation_path)):
                if f"{generation}/{index}" == keep:
                    continue
                shutil.rmtree(os.path.join(generation_path, index), ignore_errors=True)
            if not os.listdir(generation_path):
                os.rmdir(generation_path)

    def record_generation(self):
        """Snapshot this generation's organisms after they have been scored.

        Called from score_fitness, which is where every organism receives its
        score and fitness and which runs exactly once per generation. Only
        scalars are kept -- holding the organisms would pin the state a
        sequential run works to release.
        """
        generation = len(self.generation_history) + 1
        self.generation_history.append([
            {
                "generation": generation,
                "organism_index": i,
                "score": organism.score,
                "fitness": organism.fitness,
                "dna": dna2str(organism.dna, organism.parameters),
                "parameters": deepcopy(organism.parameters),
                # The organism's real folder, not one derived from its position.
                # A carried-forward elite does not re-run, so its state stays in
                # the generation that produced it -- recording 2/0 for it would
                # point at a folder nothing ever wrote.
                "folder": self.relative_folder(organism),
            }
            for i, organism in enumerate(self.population)
        ])

    def relative_folder(self, organism):
        """An organism's folder, relative to the run folder.

        Relative so the summary survives the run tree being moved or copied.

        Args:
            organism: The organism to locate.

        Returns:
            A path like "1/3", or None if it has no folder yet.
        """
        if not organism.folder:
            return None
        return os.path.relpath(organism.folder, self.run_folder()).replace(os.sep, "/")

    def run_folder(self):
        """The version-level folder holding this run's summary and generations."""
        return f"{self.directory}/{self.run_name}/{self.run_version}"

    def organism_folder(self, generation, organism_index):
        """Where one organism's state lives.

        Generation and index are both in the path, so nothing a later generation
        writes can land on an earlier one's folder. Numbering by population
        position alone let a carried-forward elite keep a path that the organism
        at that position in the next generation then overwrote, and the run
        published the wrong organism's data under the winner's score.

        Args:
            generation: 1-based generation number.
            organism_index: Position in that generation's population.

        Returns:
            Path to the organism's folder.
        """
        return f"{self.run_folder()}/{generation}/{organism_index}"

    def build_run_summary(self, best_entry):
        """Everything a caller needs to record the run, without tuner internals.

        Args:
            best_entry: The winning organism's generation_history entry.

        Returns:
            The dict written to run_summary.json and returned from run().
        """
        return {
            "run_name": self.run_name,
            "version": self.run_version,
            "started_at": self.started_at,
            "finished_at": int(time.time()),
            "pop_size": self.population_size,
            "generations": self.generations,
            "goal": self.goal,
            "best": {
                "generation": best_entry["generation"],
                "organism_index": best_entry["organism_index"],
                "score": best_entry["score"],
                "folder": best_entry["folder"],
            },
            "metrics": self.metrics,
            "generations_detail": [
                {"generation": n + 1, "organisms": organisms}
                for n, organisms in enumerate(self.generation_history)
            ],
        }

    def find_best_entry(self):
        """The generation_history entry with the best score, across all generations.

        Across all of them, not just the last: a non-elitist generation can end
        worse than one before it, and the run should not return something it
        already beat.

        Returns:
            The winning entry, or None if nothing was recorded.
        """
        entries = [e for gen in self.generation_history for e in gen
                   if e["score"] is not None]
        if not entries:
            return None
        pick = max if self.goal == "max" else min
        return pick(entries, key=lambda e: e["score"])

    def publish_run_summary(self, best_entry):
        """Write run_summary.json and the score scatter to the version folder.

        The winner is a pointer here, not a copy: it stays in its own
        generation/organism folder and the summary says where.

        Args:
            best_entry: The winning organism's generation_history entry.

        Returns:
            The summary dict.
        """
        summary = self.build_run_summary(best_entry)
        os.makedirs(self.run_folder(), exist_ok=True)
        with open(f"{self.run_folder()}/{RUN_SUMMARY_FILE}", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        self.publish_generation_scores()
        return summary

    def publish_generation_scores(self):
        """Draw the run's score scatter beside its summary.

        A chart is a report, not a result: a run that has just spent hours
        computing should not lose its output because a plotting backend is
        unhappy. Failures are logged and swallowed.

        Returns:
            Path to the image, or None if it was not drawn.
        """
        try:
            return plot_generation_scores(
                self.generation_history, self.run_folder(), goal=self.goal,
                title=f"{self.run_name} / {self.run_version}")
        except Exception:
            _log.exception("could not draw the generation score chart")
            return None


def find_checkpoint_prefixes(population):
    """Return the DNA prefixes worth caching, and how many organisms share each.

    A prefix earns a checkpoint only where the population actually forks after
    it: more than one organism shares it, and they do not all agree on the next
    gene. Every shared prefix would mean writing the whole state once per gene;
    only the fork points are ever read back.

    Args:
        population: The organisms about to run.

    Returns:
        {dna prefix: number of organisms sharing it}.
    """
    if len(population) < 2:
        return {}
    n_genes = min(len(organism.dna) for organism in population)
    prefixes = [
        [dna2str(o.dna[: i + 1], o.parameters) for i in range(n_genes)]
        for o in population
    ]

    checkpoints = {}
    # The last gene has nothing after it to fork, so it is never a checkpoint.
    for i in range(n_genes - 1):
        sharers = {}  # prefix -> [organisms sharing it, prefixes they go on to]
        for organism_prefixes in prefixes:
            n_sharing, next_prefixes = sharers.setdefault(
                organism_prefixes[i], [0, set()]
            )
            sharers[organism_prefixes[i]][0] = n_sharing + 1
            next_prefixes.add(organism_prefixes[i + 1])
        for prefix, (n_sharing, next_prefixes) in sharers.items():
            if n_sharing > 1 and len(next_prefixes) > 1:
                checkpoints[prefix] = n_sharing
    return checkpoints


def natural_selection(
        population,
        survival_variation=0.1,
        survival_percentage=0.5,
        min_probability=0.01
):
    sv = survival_variation
    prob_dist = np.array([p.fitness * (1 + random.uniform(-sv, sv)) for p in population])
    prob_dist = np.clip(prob_dist, min_probability, None)
    prob_dist /= sum(prob_dist)
    n_survivors = int(survival_percentage * len(population))

    return np.random.choice(population, size=n_survivors, replace=False, p=prob_dist)


def mutate(organism, model_space, hyperparam_space, func_prob, nuc_prob):
    """
    mutates the genes of an organism, making it make different decisions

    :param organism: object of class Organism
    :param model_space:
    :param hyperparam_space: all possible values of parameters to plug into dna
    :param func_prob: probability of mutation of the gene's function
    :param nuc_prob: probability of mutation of each of gene's nucleotides
    :param max_disc_shift: max discrete shift = the max change in index of the nucleotide option if options are discrete
    :param max_cont_shift: max continuous shift = the max change in the value of a nucleotide if options are continuous
    max_cont_shift is a percentage of the range from min to max value of nucleotide
    """
    # modify organism.dna
    for i, gene in enumerate(organism.dna):
        gene_space = model_space[i]
        if len(gene_space) > 1 and random.random() <= func_prob:
            available_genes = [g for g in gene_space if g != organism.dna[i]]
            organism.dna[i] = choose_gene(available_genes)
            continue

    # modify organism.parameters
    for hp, val in organism.parameters.items():
        if random.random() <= nuc_prob:
            organism.parameters[hp] = hyperparam_space[hp].mutate(val)


def choose_dna(dna_space):
    dna = []
    for gene_space in dna_space:
        # choose a random function from supplied choices
        gene = choose_gene(gene_space)
        dna.append(gene)
    return dna


def choose_gene(gene_space):
    return deepcopy(random.choice(gene_space))


def choose_hyperparams(hyperparam_space):
    return {key: val.sample() for key, val in hyperparam_space.items()}


def generate_stratified_folds(data, y_col, n_splits=5):
    while data is None:
        yield None, None, None, None  # Assumed to be supplied within a step in the model space
    x = data.drop(y_col, axis=1)
    y = data[y_col]
    skf = StratifiedKFold(n_splits=n_splits)
    splits = list(skf.split(x, y))
    i = 0
    while True:
        train_idx, test_idx = splits[i % n_splits]
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        yield x_train, x_test, y_train, y_test
        i += 1


"""
Future TODO:
    Make ability to add steps before splitting data for ease of use with new data (Or user supplied generators?)
    let mutation magnitude & probability adjust during the run
"""
