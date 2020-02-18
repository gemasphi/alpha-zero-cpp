#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Game.h"
#include "Tictactoe.h"


namespace py = pybind11;

PYBIND11_MODULE(game_py, m) {
    py::class_<TicTacToe>(m, "tictactoe")
        .def(py::init<>())
        .def("getBoardSize", &TicTacToe::getBoardSize)
        .def("getActionSize", &TicTacToe::getActionSize)
        .def("getInputPlanes", &TicTacToe::getInputPlanes)
        .def("getOutputPlanes", &TicTacToe::getOutputPlanes);
   
}
