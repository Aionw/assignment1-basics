#include <map>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

#include "tokenizer.h"

namespace nb = nanobind;

NB_MODULE(ctokenizer, m) {
    nb::class_<Tokenizer::BPETrainer>(m, "BPETrainer")
        .def(nb::init<size_t, const std::string&, const std::vector<std::string>&, size_t>())
        .def("train", &Tokenizer::BPETrainer::train)
        .def("vocab", [](const Tokenizer::BPETrainer& self) -> nb::dict {
            nb::dict result;
            for (const auto& [k, v] : self.vocab()) {
                result[nb::int_(k)] = nb::bytes(v.data(), v.size());
            }
            return result;
        })
        .def("merges", [](const Tokenizer::BPETrainer& self) -> nb::list {
            nb::list result;
            for (const auto& [left, right] : self.merges()) {
                result.append(nb::make_tuple(
                    nb::bytes(left.data(), left.size()),
                    nb::bytes(right.data(), right.size())
                ));
            }
            return result;
        });
}
