import asyncio
import unittest
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import MagicMock, patch

from ezmt.organism import (
    DISPATCH_AWAIT,
    DISPATCH_PARENT,
    DISPATCH_SUBPROCESS,
    DISPATCH_THREAD,
    Organism,
    copies_arguments,
    resolve_dispatch,
)


def sync_func(x):
    return x


async def async_func(x):
    return x


class TestResolveDispatch(unittest.TestCase):

    def test_coroutine_is_awaited_in_the_calling_process(self):
        self.assertEqual(resolve_dispatch(async_func), DISPATCH_AWAIT)

    def test_coroutine_wins_over_run_in_parent_process(self):
        # An async gene is awaited regardless of the flag, so both agree it stays
        # in the calling process. Order matters only for the returned label.
        self.assertEqual(
            resolve_dispatch(async_func, run_in_parent_process=True), DISPATCH_AWAIT
        )

    def test_run_in_parent_process_keeps_a_sync_gene_inline(self):
        self.assertEqual(
            resolve_dispatch(sync_func, run_in_parent_process=True), DISPATCH_PARENT
        )

    def test_plain_sync_gene_goes_to_the_pool(self):
        self.assertEqual(resolve_dispatch(sync_func), DISPATCH_SUBPROCESS)

    def test_plain_sync_gene_without_a_pool_uses_a_thread(self):
        self.assertEqual(
            resolve_dispatch(sync_func, has_pool=False), DISPATCH_THREAD
        )


class TestCopiesArguments(unittest.TestCase):
    """Only a process hop copies arguments; everything else shares by reference."""

    def test_pooled_sync_gene_copies(self):
        self.assertTrue(copies_arguments(sync_func))

    def test_async_gene_does_not_copy(self):
        self.assertFalse(copies_arguments(async_func))

    def test_parent_process_gene_does_not_copy(self):
        self.assertFalse(copies_arguments(sync_func, run_in_parent_process=True))

    def test_threaded_gene_does_not_copy(self):
        self.assertFalse(copies_arguments(sync_func, has_pool=False))


class TestRunGeneUsesResolveDispatch(unittest.IsolatedAsyncioTestCase):
    """run_gene must route through resolve_dispatch, not its own copy of the rule."""

    def _organism(self, func, run_in_parent_process=False):
        # outputs is a list because validate_config normalises it to one before
        # run_gene ever sees it; _update_state relies on that.
        gene = {
            "name": "g",
            "train": {
                "func": func,
                "args": ["value"],
                "kwargs": {},
                "outputs": ["value"],
                "run_in_parent_process": run_in_parent_process,
            },
        }
        return Organism("test", [gene], {})

    async def test_async_gene_is_awaited_without_touching_the_pool(self):
        organism = self._organism(async_func)
        pool = MagicMock()
        state = await organism.run_gene("train", 0, {"value": 7}, pool=pool)
        self.assertEqual(state["value"], 7)
        pool.submit.assert_not_called()

    async def test_parent_process_gene_does_not_touch_the_pool(self):
        organism = self._organism(sync_func, run_in_parent_process=True)
        pool = MagicMock()
        state = await organism.run_gene("train", 0, {"value": 7}, pool=pool)
        self.assertEqual(state["value"], 7)
        pool.submit.assert_not_called()

    async def test_dispatch_decision_comes_from_resolve_dispatch(self):
        # Forcing the predicate to report PARENT must change run_gene's behaviour,
        # proving it is the single source of truth rather than a parallel copy.
        organism = self._organism(sync_func)
        pool = MagicMock()
        with patch(
            "ezmt.organism.resolve_dispatch", return_value=DISPATCH_PARENT
        ) as resolver:
            state = await organism.run_gene("train", 0, {"value": 7}, pool=pool)
        resolver.assert_called_once()
        self.assertEqual(state["value"], 7)
        pool.submit.assert_not_called()

    async def test_sync_gene_actually_runs_in_a_subprocess(self):
        organism = self._organism(sync_func)
        with ProcessPoolExecutor(1) as pool:
            state = await organism.run_gene("train", 0, {"value": 7}, pool=pool)
        self.assertEqual(state["value"], 7)


if __name__ == "__main__":
    unittest.main()
