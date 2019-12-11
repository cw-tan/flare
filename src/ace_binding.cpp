#include "ace.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

// Define spherical harmonics class.
class SphericalHarmonics{
    public:
        double x, y, z;
        int lmax;
        std::vector<double> Y, Yx, Yy, Yz;

        SphericalHarmonics(double x, double y, double z, int lmax);
};

SphericalHarmonics :: SphericalHarmonics(double x, double y, double z,
                                         int lmax){
    int number_of_harmonics = (lmax + 1) * (lmax + 1);

    this->x = x;
    this->y = y;
    this->z = z;
    this->lmax = lmax;

    // Initialize spherical harmonic vectors.
    Y = std::vector<double>(number_of_harmonics, 0);
    Yx = std::vector<double>(number_of_harmonics, 0);
    Yy = std::vector<double>(number_of_harmonics, 0);
    Yz = std::vector<double>(number_of_harmonics, 0);

    get_Y(Y, Yx, Yy, Yz, x, y, z, lmax);
};

PYBIND11_MODULE(ace, m){
    // Bind the spherical harmonics class.
    py::class_<SphericalHarmonics>(m, "SphericalHarmonics")
        .def(py::init<double, double, double, int>())
        // Make attributes accessible.
        .def_readwrite("x", &SphericalHarmonics::x)
        .def_readwrite("y", &SphericalHarmonics::y)
        .def_readwrite("z", &SphericalHarmonics::z)
        .def_readwrite("lmax", &SphericalHarmonics::lmax)
        .def_readwrite("Y", &SphericalHarmonics::Y)
        .def_readwrite("Yx", &SphericalHarmonics::Yx)
        .def_readwrite("Yy", &SphericalHarmonics::Yy)
        .def_readwrite("Yz", &SphericalHarmonics::Yz);
    
    // Bind the structure class.
    py::class_<Structure>(m, "Structure")
        .def(py::init<const std::vector<double> &,
                      const std::vector<double> &,
                      const std::vector<double> &,
                      const std::vector<double> &,
                      const std::vector<double> &,
                      const std::vector<double> &,
                      const std::vector<int> &>())
        .def_readwrite("xs", &Structure::xs)
        .def_readwrite("ys", &Structure::ys)
        .def_readwrite("zs", &Structure::zs)
        .def_readwrite("vec1", &Structure::vec1)
        .def_readwrite("vec2", &Structure::vec2)
        .def_readwrite("vec3", &Structure::vec3)
        .def_readwrite("species", &Structure::species);
}
