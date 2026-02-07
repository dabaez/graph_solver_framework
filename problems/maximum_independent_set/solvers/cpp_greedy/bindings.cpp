#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

std::vector<int> solve_mis_greedy(int n, const std::vector<std::pair<int, int>>& edges);

namespace py = pybind11;

PYBIND11_MODULE(mis_greedy_cpp, m) {
    m.doc() = "Greedy Maximum Independent Set Solver";
    m.def("solve", &solve_mis_greedy, "Solve Maximum Independent Set using a greedy algorithm",
          py::arg("num_nodes"), py::arg("edges"));
}
