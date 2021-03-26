#include <iostream>
#include <pybind11/embed.h>
// #include <pybind11/numpy.h>
// #include <pybind11/eigen.h>
#include <dart/dart.hpp>
namespace py = pybind11;
int main(int argc,char** argv)
{
    Py_Initialize();
	// py::str module_dir = "../";
	// py::module::import("sys").attr("path").attr("insert")(0, module_dir);
 //    py::object nn_module = py::module::import("Model").attr("ActorCriticNN")(163, 35).attr("cuda")();
 //    nn_module.attr("load")("../current_0.pt");


    py::object nn_module = py::module::import("torch").attr("load")("../current_0.pt");

    auto world= new dart::simulation::World();
}
