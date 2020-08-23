float toRange( float originalStart, float originalEnd,  // original range
               float newStart,      float newEnd,       // desired range
               float value)                             // value to convert
{
    float originalDiff = originalEnd - originalStart;
    float newDiff = newEnd - newStart;
    float ratio = newDiff / originalDiff;
    float newProduct = value * ratio;
    float finalValue = newProduct + newStart;
    return finalValue; 
}

float toRange2( float OLD_MIN, float OLD_MAX,  // original range
                float NEW_MIN, float NEW_MAX,  // desired range
                float value)                 // value to convert
{
    float old_range = (value - OLD_MIN) / (OLD_MAX - OLD_MIN);
    return ((NEW_MAX - NEW_MIN) * old_range) + NEW_MIN;
}

// returns the 2D distance from camera to a position in worldSpace
float get_camera_distance_to_pos2d(float2 position_xy)
{
    return distance(inverseWorldMatrix[3].xy, position_xy);
}

// returns the 3D distance from camera to a position in worldSpace
float get_camera_distance_to_pos3d(float3 position_xyz)
{
    return distance(inverseWorldMatrix[3].xyz, position_xyz);
}

// returns 1 if facing forward, 0 if facing up/down
float get_camera_angle_vertical()
{
    return viewMatrix[2].y;
}

float4x4 rotation_matrix_4x4(float3 axis, float4 d)
{
    float cx, cy, cz, sx, sy, sz;

    sincos(axis.x, sx, cx);
    sincos(axis.y, sy, cy);
    sincos(axis.z, sz, cz);	

    return float4x4( cy*cz,     -sz,    sy, d.x,
                     sz,      cx*cz,   -sx, d.y,
                    -sy,         sx, cx*cy, d.z,
                      0,          0,     0, d.w );
}

//
// tangent 
//

// input :: (float4) vertex.tangent
float3 setup_tangent(float4 vertex_tangent)
{
    return (vertex_tangent.zxy * 0.00787401572 - 1.0) * (vertex_tangent.w * 0.00392156886 + 0.752941191);
}

// transforms vertex tangent to clipspace (uses world/view/projection matrices)
float4 transform_tangent_clipspace_wvp(float3 setup_tangent)
{
    float4 tan_proj;

    tan_proj = mul(setup_tangent.yzx,  worldMatrix);
    tan_proj = mul(tan_proj, viewMatrix);
    tan_proj = mul(tan_proj, projectionMatrix);

    return float4(1, -1, 1, 1) * clipSpaceLookupScale * tan_proj;
}

// transforms vertex tangent to clipspace (uses worldViewProjection matrix)
float4 transform_tangent_clipspace(float3 setup_tangent)
{
    float4 tan_proj = mul(setup_tangent.yzx,  worldViewProjectionMatrix);

    return float4(1, -1, 1, 1) * clipSpaceLookupScale * tan_proj;
}

//
// binormal
//

// returns the binormal :: input :: (float4) vertex.normal
float3 setup_binormal(float4 vertex_normal)
{
    return (vertex_normal.yzx * 0.00787401572 - 1.0) * (vertex_normal.w * 0.00392156886 + 0.752941191);
}

// transforms binormal to clipspace (uses world/view/projection matrices)
float4 transform_binormal_clipspace_wvp(float3 setup_tangent, float3 setup_binormal)
{
    float4 binormal_proj;

    binormal_proj = mul(setup_tangent.zxy * setup_binormal.yzx - (setup_tangent * setup_binormal), worldMatrix);
    binormal_proj = mul(binormal_proj, viewMatrix);
    binormal_proj = mul(binormal_proj, projectionMatrix);
    
    return float4(-1, 1, -1, -1) * clipSpaceLookupScale * binormal_proj;
}

// transforms binormal to clipspace (uses worldViewProjection matrix)
float4 transform_binormal_clipspace(float3 setup_tangent, float3 setup_binormal)
{
    float4 binormal_proj = mul(setup_tangent.zxy * setup_binormal.yzx - (setup_tangent * setup_binormal), worldViewProjectionMatrix);

    return float4(-1, 1, -1, -1) * clipSpaceLookupScale * binormal_proj;
}