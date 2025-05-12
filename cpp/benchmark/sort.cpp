#include <benchmark/benchmark.h>
#include <mpi.h>

#include <dolfinx/mesh/generation.h>

static void BM_create_box(benchmark::State& state)
{
  spdlog::set_level(spdlog::level::warn);
  for (auto _ : state)
  {
    // This code gets timed
    state.PauseTiming();
    auto n = state.range(0);
    state.ResumeTiming();
    dolfinx::mesh::create_box(MPI_COMM_SELF, {{{0, 0, 0}, {1, 1, 1}}},
                              {n, n, n}, dolfinx::mesh::CellType::tetrahedron);
  }
}

BENCHMARK(BM_create_box)
    ->Args({10})
    ->Args({50})
    ->Args({100})
    // ->Args({200})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(5);

int main(int argc, char** argv)
{
  char arg0_default[] = "benchmark";
  char* args_default = arg0_default;
  if (!argv)
  {
    argc = 1;
    argv = &args_default;
  }
  MPI_Init(&argc, &argv);
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();

  MPI_Finalize();
  return 0;
}
