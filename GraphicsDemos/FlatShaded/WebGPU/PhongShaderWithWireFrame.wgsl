@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<uniform> material : Material;

struct Uniforms
{
    MVP : mat4x4<f32>,
    M : mat4x4<f32>,
    normal_matrix : mat3x3<f32>,
};


struct Material {
    objectColour : vec3<f32>,     // padded to 16 bytes
    lightColour  : vec3<f32>,     // padded to 16 bytes
    lightPos     : vec3<f32>,     // padded to 16 bytes
    viewPos      : vec3<f32>,     // padded to 16 bytes
    // Note: params.x is specularStrength, params.y is shininess
    // params.z is line_width, params.w is is_wireframe
    // This is to avoid padding issues with f32
    params : vec4<f32>,
};

// Define the constant array at the module scope
const BARYCENTRIC_COORDS : array<vec3<f32>, 3> = array<vec3<f32>, 3>(
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0)
);



struct VertexIn {
    @builtin(vertex_index) vid : u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};
struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) barycentric: vec3<f32>,
    @location(3) fragPos : vec3<f32>,
};


@vertex
fn vertex_main(input: VertexIn) -> VertexOut
{
    var output: VertexOut;
    output.position = uniforms.MVP * vec4<f32>(input.position, 1.0);


    output.fragPos = (uniforms.M * vec4<f32>(input.position,1.0)).xyz;
    output.normal = uniforms.normal_matrix * input.normal;
    output.uv = input.uv;

    let selector = input.vid % 3u;
    output.barycentric = BARYCENTRIC_COORDS[selector];

    return output;
}

// This function calculates how close a fragment is to a triangle edge
fn edge_factor(bary: vec3<f32>, width: f32) -> f32
    {
    // Get screen-space derivatives of the barycentric coordinates
    let d = fwidth(bary);
    // Calculate smoothed distance to the nearest edge
    let a3 = smoothstep(vec3<f32>(0.0), d * width, bary);
    // The minimum value is the distance to the closest edge
    return min(min(a3.x, a3.y), a3.z);
}


@fragment
fn fragment_main(in: VertexOut) -> @location(0) vec4<f32>
{
// --- Lighting (1:1 with GLSL) ---
let ambientStrength = 0.1;
let ambient = ambientStrength * material.lightColour;

let N = normalize(in.normal);
let L = normalize(material.lightPos - in.fragPos);
let diff = max(dot(N, L), 0.0);
let diffuse = diff * material.lightColour;

let V = normalize(material.viewPos - in.fragPos);
let H = normalize(L + V);
let spec = pow(max(dot(N, H), 0.0), material.params.y);
let specular = material.params.x * spec * material.lightColour;

let surface_color = vec4<f32>((ambient + diffuse + specular) * material.objectColour,1.0);
if (material.params.w > 0.5)
{
    let factor = edge_factor(in.barycentric, material.params.z * 10.0 + 1.0);

    // The 'factor' is 0.0 on an edge and 1.0 in the middle of a triangle.
    // We can use its inverse for the alpha channel.
    let alpha = 1.0 - factor;

    // Return the surface color, but with a calculated alpha.
    // This makes the center of the triangle transparent.
    return vec4<f32>(surface_color.rgb, alpha);
}
else
{
    return surface_color;
}
}
