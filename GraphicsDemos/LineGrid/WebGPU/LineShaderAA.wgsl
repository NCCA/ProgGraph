@group(0) @binding(0) var<uniform> uniforms : Uniforms;
struct Uniforms
{
    MVP : mat4x4<f32>,
};

struct VertexIn {
    @location(0) position: vec3<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) screen_pos: vec2<f32>,
};

@vertex
fn vertex_main(input: VertexIn) -> VertexOut {
    var output: VertexOut;
    output.position = uniforms.MVP * vec4<f32>(input.position, 1.0);
    // Convert to screen space for interpolation
    let clip_pos = output.position.xy / output.position.w;
    output.screen_pos = clip_pos * 0.5 + 0.5;
    return output;
}
@fragment
fn fragment_main(@location(0) screen_pos: vec2<f32>) -> @location(0) vec4<f32> {
    // Calculate screen-space gradients for anti-aliasing
    let grad_x = dpdx(screen_pos).xy;
    let grad_y = dpdy(screen_pos).xy;
    let grad_len = max(length(grad_x), length(grad_y));

    // Create anti-aliased grid lines
    let grid_freq = 30.0; // Number of grid divisions
    let grid_coords = screen_pos * grid_freq;

    // Distance to nearest grid line
    var dist_to_line = min(fract(grid_coords.x), 1.0 - fract(grid_coords.x));
    dist_to_line = min(dist_to_line, min(fract(grid_coords.y), 1.0 - fract(grid_coords.y)));

    // Calculate anti-aliased alpha based on distance and screen-space derivatives
    let line_width = grad_len * 2.0; // Base line width in screen space
    let coverage = 1.0 - smoothstep(0.0, line_width, dist_to_line);

    // Apply additional smoothing based on gradients
    let smooth_width = grad_len * 4.0;
    let smooth_coverage = 1.0 - smoothstep(line_width, smooth_width, dist_to_line);

    // Blend sharp and smooth coverage for best results
    let final_alpha = mix(coverage, smooth_coverage, 0.5);

    return vec4<f32>(1.0, 1.0, 1.0, final_alpha);
}
