// Verilator forward-pass co-simulation testbench for watch_grid_4x4.sv.
//
// Build after installing Verilator:
//   verilator --cc --exe --build --top-module watch_grid_4x4 \
//     watch_grid_4x4.sv tb_watch_grid_co_sim.cpp
// Run:
//   ./obj_dir/Vwatch_grid_4x4 watch_grid_vectors.txt

#include "Vwatch_grid_4x4.h"
#include "verilated.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Vectors {
    std::int8_t activation[4]{};
    std::uint8_t weight_code[4][4]{};
    std::int32_t expected[4]{};
};

template <typename T>
void require_read(std::istream& input, T& value, const char* field) {
    if (!(input >> value)) {
        throw std::runtime_error(std::string("missing vector field: ") + field);
    }
}

Vectors read_vectors(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open vector file: " + path);
    }
    std::string header;
    int rows = 0;
    int columns = 0;
    require_read(input, header, "header");
    require_read(input, rows, "rows");
    require_read(input, columns, "columns");
    if (header != "WATCH_GRID_COSIM_V1" || rows != 4 || columns != 4) {
        throw std::runtime_error("unsupported vector header or dimensions");
    }

    Vectors vectors;
    std::string field;
    require_read(input, field, "activations label");
    if (field != "activations") throw std::runtime_error("expected activations field");
    for (auto& value : vectors.activation) {
        int raw = 0;
        require_read(input, raw, "activation");
        value = static_cast<std::int8_t>(raw);
    }

    require_read(input, field, "weight_codes label");
    if (field != "weight_codes") throw std::runtime_error("expected weight_codes field");
    for (auto& row : vectors.weight_code) {
        for (auto& value : row) {
            int raw = 0;
            require_read(input, raw, "weight code");
            value = static_cast<std::uint8_t>(raw);
        }
    }

    require_read(input, field, "expected label");
    if (field != "expected") throw std::runtime_error("expected output field");
    for (auto& value : vectors.expected) require_read(input, value, "expected output");
    return vectors;
}

void tick(Vwatch_grid_4x4* dut) {
    dut->clk = 0;
    dut->eval();
    dut->clk = 1;
    dut->eval();
}

int run(const Vectors& vectors, bool print_result) {
    VerilatedContext context;
    context.commandArgs(0, nullptr);
    auto* dut = new Vwatch_grid_4x4{&context};
    dut->reset_n = 0;
    dut->FW_BW_SELECT = 1;
    for (int row = 0; row < 4; ++row) {
        dut->grad_y_in[row] = 0;
        for (int column = 0; column < 4; ++column) {
            dut->weight_code[row][column] = vectors.weight_code[row][column];
            dut->latent_weight[row][column] = 0;
        }
    }
    for (int column = 0; column < 4; ++column) dut->activation_in[column] = vectors.activation[column];
    tick(dut);
    dut->reset_n = 1;
    for (int cycle = 0; cycle < 4; ++cycle) tick(dut);

    int mismatches = 0;
    for (int row = 0; row < 4; ++row) {
        const auto actual = static_cast<std::int16_t>(dut->output_y[row]);
        if (actual != vectors.expected[row]) {
            ++mismatches;
            std::cerr << "row " << row << ": expected " << vectors.expected[row]
                      << ", actual " << actual << '\n';
        }
        if (print_result) std::cout << "RESULT " << row << ' ' << actual << '\n';
    }
    delete dut;
    return mismatches;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string path = argc > 1 ? argv[1] : "watch_grid_vectors.txt";
        const auto vectors = read_vectors(path);
        const int mismatches = run(vectors, true);
        if (mismatches == 0) {
            std::cout << "WATCH_GRID_COSIM PASS bit-exact=4/4\n";
            return 0;
        }
        std::cerr << "WATCH_GRID_COSIM FAIL mismatches=" << mismatches << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "WATCH_GRID_COSIM ERROR " << error.what() << '\n';
        return 2;
    }
}