#define PC
#define IS_VERTEX_SHADER 0
#define IS_PIXEL_SHADER 1

// no support for custom code constants when using fullbright
//#define USE_CUSTOM_CONSTANTS

#include <shader_vars.h>
#include <lib/utility.hlsl>
#include <lib/ocean_utility.hlsl>

struct PixelInput
{
    float4 position  : POSITION;
	float3 normal 	 : NORMAL;
	float3 uv 		 : TEXCOORD0;
	float3 tangent 	 : TEXCOORD1;
	float3 bitangent : TEXCOORD2;
	float3 worldPos  : TEXCOORD3;
	float2 wind 	 : TEXCOORD4;
};

float4 ps_main( const PixelInput pixel ) : COLOR
{
	// both vs / ps
	const float4 _WaveAmplitude = float4(0.72, 0.4, 0.46, 1.93);
	  	  float4 _WavesIntensity = float4(0.28, 1.95, 1, 1.9);
	  	  float4 _WavesNoise = float4(0.05, 0.45, 1.18, 0.28);

	const float  _NormalIntensity = 0.9549;
	const float  _Distortion = 0.4;

	float  timer 		= pixel.uv.z;
	float2 timedWindDir = pixel.wind.xy * timer;
	float3 surfacePosition = pixel.worldPos;

	//wave normal
	half3 normal = ComputeNormal(normalMapSampler, surfacePosition.xz, pixel.uv, pixel.normal, pixel.tangent, pixel.bitangent, _WavesNoise, _WavesIntensity, timedWindDir);
	normal = normalize(lerp(pixel.normal, normalize(normal), _NormalIntensity));

	// rotate reflection vector by 180 deg
	float3 reflectVec = normalize(reflect(pixel.worldPos.xyz, normal.xzy * _Distortion));

	float4x4 rot_mat = rotation_matrix_4x4(float3(0.0, 0.0, M_PI), float4(0.0, 0.0, 0.0, 1.0));
	reflectVec = mul(reflectVec, rot_mat);

	return float4(texCUBE(skyMapSampler, reflectVec).rgb, 1.0f);
}
