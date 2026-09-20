// C ABI for direct Python ctypes hardware-in-the-loop calls.
// Build this translation unit together with Verilator's generated model and
// load the resulting shared library from watch_grid_hil.py.

#include "Vwatch_grid_4x4.h"
#include "verilated.h"

#include <cstdint>

extern "C" int watch_grid_run(
    const std::int8_t* activation,
    const std::uint8_t* weight_code,
    std::int32_t* output
) {
    VerilatedContext context;
    context.commandArgs(0, nullptr);
    auto* dut = new Vwatch_grid_4x4{&context};
    dut->FW_BW_SELECT = 1;
    dut->reset_n = 0;
    for (int row = 0; row < 4; ++row) {
        dut->grad_y_in[row] = 0;
        for (int column = 0; column < 4; ++column) {
            dut->weight_code[row][column] = weight_code[row * 4 + column];
            dut->latent_weight[row][column] = 0;
        }
    }
    for (int column = 0; column < 4; ++column) dut->activation_in[column] = activation[column];

    auto tick = [dut]() {
        dut->clk = 0;
        dut->eval();
        dut->clk = 1;
        dut->eval();
    };
    tick();
    dut->reset_n = 1;
    for (int cycle = 0; cycle < 4; ++cycle) tick();
    for (int row = 0; row < 4; ++row)
        output[row] = static_cast<std::int16_t>(dut->output_y[row]);
    delete dut;
    return 0;
}