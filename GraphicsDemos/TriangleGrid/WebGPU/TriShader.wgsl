@group(0) @binding(0) var<uniform> uniforms : Uniforms;
struct Uniforms
{
    MVP : mat4x4<f32>,
};




struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};
struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
};


@vertex
fn vertex_main(input: VertexIn) -> VertexOut {
    var output: VertexOut;
    output.position = uniforms.MVP * vec4<f32>(input.position, 1.0);
    output.normal = input.normal;
    output.uv = input.uv;

    return output;
}


@fragment
fn fragment_main(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.uv.x, in.uv.y, 0.0, 1.0);

}
