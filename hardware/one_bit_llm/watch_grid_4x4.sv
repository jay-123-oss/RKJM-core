// 2D Watch Grid hybrid PE array.
//
// The default instance is a 4x4 systolic array.  weight_code uses the same
// two-bit encoding as the Python model: 00=zero, 01=+1, 11=-1.  The FP32
// update path is represented as signed Q2.(FP_WIDTH-2) arithmetic so this
// module remains synthesizable without binding to a vendor floating-point IP
// core.  A production FP32 implementation can replace the two marked
// multiply/add operators with the target library's IEEE-754 unit while
// retaining the FW_BW_SELECT and register interfaces.

module watch_grid_4x4 #(
    parameter int ROWS = 4,
    parameter int COLS = 4,
    parameter int ACC_WIDTH = 16,
    parameter int ACT_WIDTH = 8,
    parameter int FP_WIDTH = 32,
    parameter int CYCLE_WIDTH = (COLS <= 1) ? 1 : $clog2(COLS)
) (
    input  logic clk,
    input  logic reset_n,
    input  logic FW_BW_SELECT, // 1: ternary forward, 0: latent gradient update
    input  logic signed [ACT_WIDTH-1:0] activation_in [0:COLS-1],
    input  logic signed [ACT_WIDTH-1:0] grad_y_in [0:ROWS-1],
    input  logic [1:0] weight_code [0:ROWS-1][0:COLS-1],
    input  logic signed [FP_WIDTH-1:0] latent_weight [0:ROWS-1][0:COLS-1],
    output logic signed [ACC_WIDTH-1:0] output_y [0:ROWS-1],
    output logic signed [FP_WIDTH-1:0] grad_w_out [0:ROWS-1][0:COLS-1]
);

    localparam logic signed [FP_WIDTH-1:0] ONE_FP =
        ({{(FP_WIDTH-1){1'b0}}, 1'b1} <<< (FP_WIDTH - 2));

    logic [CYCLE_WIDTH-1:0] cycle_idx;
    logic [ACC_WIDTH-1:0] sum_reg [0:ROWS-1];
    logic [ACC_WIDTH-1:0] carry_reg [0:ROWS-1];
    logic [ACC_WIDTH-1:0] sum_next [0:ROWS-1];
    logic [ACC_WIDTH-1:0] carry_next [0:ROWS-1];
    logic signed [1:0] ternary_weight [0:ROWS-1];
    logic signed [ACT_WIDTH:0] product_value [0:ROWS-1];
    logic [ACC_WIDTH-1:0] product_bits [0:ROWS-1];
    logic signed [FP_WIDTH-1:0] grad_next [0:ROWS-1][0:COLS-1];
    logic signed [FP_WIDTH-1:0] gradient_product [0:ROWS-1];

    integer row;
    integer col;

    // Combinational forward datapath: decode one vertical weight column,
    // form signed products, and reduce P/S/C with the CSA equations.
    always_comb begin
        for (row = 0; row < ROWS; row = row + 1) begin
            ternary_weight[row] = 2'sd0;
            if (weight_code[row][cycle_idx] == 2'b01)
                ternary_weight[row] = 2'sd1;
            else if (weight_code[row][cycle_idx] == 2'b11)
                ternary_weight[row] = -2'sd1;

            product_value[row] = activation_in[cycle_idx] * ternary_weight[row];
            product_bits[row] = '0;
            product_bits[row] = product_value[row];

            sum_next[row] = product_bits[row] ^ sum_reg[row] ^ carry_reg[row];
            carry_next[row] = (product_bits[row] & sum_reg[row])
                            | (sum_reg[row] & carry_reg[row])
                            | (carry_reg[row] & product_bits[row]);
        end
    end

    // Carry is unshifted while stored.  The final numerical value is
    // S + (C << 1), exactly as in the reference CSA model.
    always_comb begin
        for (row = 0; row < ROWS; row = row + 1) begin
            output_y[row] = $signed(sum_reg[row] + (carry_reg[row] << 1));
        end
    end

    // Combinational backward datapath.  ONE_FP defines the Q2.(N-2) value
    // one; the inclusive absolute-value comparison implements |W_fp| <= 1.
    always_comb begin
        for (row = 0; row < ROWS; row = row + 1) begin
            gradient_product[row] = grad_y_in[row] * activation_in[cycle_idx];
            for (col = 0; col < COLS; col = col + 1) begin
                grad_next[row][col] = grad_w_out[row][col];
                if ((latent_weight[row][col] <= ONE_FP)
                    && (latent_weight[row][col] >= -ONE_FP)) begin
                    grad_next[row][col] = grad_w_out[row][col]
                                         + gradient_product[row];
                end
            end
        end
    end

    // Sequential pipeline and mode isolation.  FW_BW_SELECT prevents the
    // inactive arithmetic domain from toggling on a given clock edge.
    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            cycle_idx <= '0;
            for (row = 0; row < ROWS; row = row + 1) begin
                sum_reg[row] <= '0;
                carry_reg[row] <= '0;
                for (col = 0; col < COLS; col = col + 1)
                    grad_w_out[row][col] <= '0;
            end
        end else if (FW_BW_SELECT) begin
            for (row = 0; row < ROWS; row = row + 1) begin
                sum_reg[row] <= sum_next[row];
                carry_reg[row] <= carry_next[row];
            end
            if (cycle_idx == COLS - 1)
                cycle_idx <= '0;
            else
                cycle_idx <= cycle_idx + 1'b1;
        end else begin
            for (row = 0; row < ROWS; row = row + 1)
                for (col = 0; col < COLS; col = col + 1)
                    grad_w_out[row][col] <= grad_next[row][col];
        end
    end

endmodule