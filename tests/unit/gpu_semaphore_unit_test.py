import asyncio
import unittest

from ezmt.model_tuner import ModelTuner


class TestGpuSemaphoreSerialisesGeneRuns(unittest.IsolatedAsyncioTestCase):
    """Genes marked gpu=True must not run concurrently.

    These use a coroutine containing a real suspension point. Today's training
    loop happens to have none, so its tasks run to completion one after another
    whatever the semaphore does -- but that is an accident of the loop's
    contents, not a guarantee. Adding any genuine await to it (pause support,
    async logging, IO) would let the tasks interleave, and the semaphore is what
    has to hold at that point.
    """

    def setUp(self):
        self.tuner = ModelTuner([], {}, generations=1, pop_size=1)

    async def _record_order(self, guard):
        order = []

        async def gene(tag):
            for _ in range(3):
                order.append(tag)
                await asyncio.sleep(0)  # a real suspension point
            return tag

        tasks = [asyncio.create_task(guard(gene(t))) for t in "AB"]
        for t in tasks:
            await t
        return "".join(order)

    async def test_guarded_gene_runs_do_not_interleave(self):
        # Act
        order = await self._record_order(self.tuner._hold_gpu_semaphore)
        # Assert
        self.assertIn(order, ("AAABBB", "BBBAAA"))

    async def test_unguarded_gene_runs_do_interleave(self):
        # Arrange - establishes the test can tell the difference, so the guarded
        # case above is not passing for some unrelated reason.
        async def no_guard(coro):
            return await coro

        # Act
        order = await self._record_order(no_guard)
        # Assert
        self.assertEqual(order, "ABABAB")

    async def test_semaphore_is_released_after_each_gene(self):
        # Arrange - a gene that raises must not leave the semaphore held, or
        # every later GPU gene would deadlock.
        async def failing_gene():
            raise RuntimeError("gene failed")

        # Act
        with self.assertRaises(RuntimeError):
            await self.tuner._hold_gpu_semaphore(failing_gene())

        # Assert
        self.assertFalse(self.tuner.gpu_semaphore.locked())

    async def test_return_value_passes_through(self):
        async def gene():
            return "result"

        self.assertEqual(await self.tuner._hold_gpu_semaphore(gene()), "result")


if __name__ == "__main__":
    unittest.main()
