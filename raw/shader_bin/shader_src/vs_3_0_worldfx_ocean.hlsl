// * xoxor4d.github.io
// * original shader : https://github.com/tuxalin/water-shader

#define PC
#define IS_VERTEX_SHADER 1
#define IS_PIXEL_SHADER 0

// use custom code constants (iw3xo-devgui)
#define USE_CUSTOM_CONSTANTS

// USE_DISPLACEMENT needs to be defined (utility.hlsl)
#define USE_DISPLACEMENT
#define HIGHTMAP_HASHSCALE .1051

#include <shader_vars.h>
#include <lib/transform.hlsl>
#include <lib/ocean_utility.hlsl>

// ------------------------------------------

struct VertexInput
{
	float4 position 	: POSITION;
	float4 color 		: COLOR;
};

struct PixelInput
{
    float4 position 	: POSITION;
	float4 color 		: COLOR;
	float3 normal 		: NORMAL;
	float3 uv 			: TEXCOORD0;
	float3 tangent 		: TEXCOORD1;
	float3 bitangent 	: TEXCOORD2;
	float3 worldPos 	: TEXCOORD3;
	float4 projPos 		: TEXCOORD4;
	float4 camPosInvDist: TEXCOORD5;
	float3 camPos 		: TEXCOORD6;
	float2 wind 		: TEXCOORD7;
};

// ------------------------------------------
// we have to create a procedural hightmap since cod4 cannot sample from textures within vertexshader

float heightmap_hash(in float2 p, in float scale)
{
    p = fmod(p, scale);

	float3 p3 = frac(float3(p.xyx) * HIGHTMAP_HASHSCALE);
    	   p3 += dot(p3, p3.yzx + 19.19f);

    return frac((p3.x + p3.y) * p3.z);
}

static const float2 add = float2(1.0, 0.0);
float heightmap( in float2 x, in float scale, in float time )
{
    x *= scale;
    x += time;
    
    float2 p = floor(x);
    float2 f = fract(x);
    
    f = f * f * (1.5 - f) * 2.0;
    
    float res = lerp(lerp( heightmap_hash(p, scale), heightmap_hash(p + add.xy, scale), f.x),
                     lerp( heightmap_hash(p + add.yx, scale), heightmap_hash(p + add.xx, scale), f.x), 
					 f.y);
    return res;
}

// ------------------------------------------

PixelInput vs_main( const VertexInput vertex ) 
{
	PixelInput pixel;

#ifdef USE_CUSTOM_CONSTANTS

	// both vs / ps
	float4 _WaveAmplitude 	= float4(1 * filterTap[0]);
	float4 _WavesIntensity 	= float4(1 * filterTap[1]);
	float4 _WavesNoise 		= float4(1 * filterTap[2]);

	// vs
	float2 _WindDirection 		= float2(filterTap[3].x, filterTap[3].y); // non-const or custom constants change behaviour
	float  _HeightIntensity 	= 1 * filterTap[3].z;
	float  _WaveAmplitudeFactor = 1 * filterTap[3].w;

	float  _WaveSteepness 		= 1 * filterTap[4].x;
	float  _TextureTiling 		= 1 * filterTap[4].y;
	float  _WaveTiling 			= 1 * filterTap[4].z;
	float  _Time 				= gameTime.w * (1 * filterTap[4].w);

	float _VisibleWaveDist 		= 1 * filterTap[5].x;
	float _HeightMapScale 		= 1 * filterTap[5].y;
	float _HeightMapScroll 		= gameTime.w * (1 * filterTap[5].z);

#else // ------------------------------------------

	// both vs / ps
	const float4 _WaveAmplitude = float4(0.72, 0.4, 0.46, 1.93);
	  	  float4 _WavesIntensity = float4(0.28, 1.95, 1, 1.9);
	  	  float4 _WavesNoise = float4(0.05, 0.45, 1.18, 0.28);

		  float2 _WindDirection = float2(-0.85, -3.1);
		  float  _HeightIntensity = 1.1;
	const float  _WaveAmplitudeFactor = 1.35;
	const float  _WaveSteepness = 0.65;
	const float  _TextureTiling = 0.95;
	const float  _WaveTiling = 0.06;
		  float  _Time = gameTime.w * 0.03;
	const float  _VisibleWaveDist = 0.2;
	const float  _HeightMapScale = 4;
		  float  _HeightMapScroll = gameTime.w * 0.15;

#endif // ifdef USE_CUSTOM_CONSTANTS

	// ------------------------------------------
	// note : this shader was based on Y up, while cod has Z up

	float4 _WorldSpaceCameraPos = inverseWorldMatrix[3]; //inverseViewMatrix[3]; (mix'n match local and modelspace camera)

	float2 windDir 	 = _WindDirection;
	float  windSpeed = length(_WindDirection);
	float  timer 	 = _Time * windSpeed * 10;

	windDir /= windSpeed;

	float3 worldPos = vertex.position.xzy;
	half3  normal 	= half3(0, 1, 0);

	float  cameraDistance = length(inverseWorldMatrix[3].xzy - worldPos) * _VisibleWaveDist;
	float2 noise 		  = GetNoise(worldPos.xz, timer * windDir * 0.5);

	half3  tangent;
	float4 waveSettings   = float4(windDir, _WaveSteepness, _WaveTiling);
	float4 waveAmplitudes = _WaveAmplitude * _WaveAmplitudeFactor;
	
	worldPos = ComputeDisplacement(worldPos, cameraDistance, noise, timer, waveSettings, waveAmplitudes, _WavesIntensity, _WavesNoise, normal, tangent);

	// add extra noise height from a heightmap
	float  heightIntensity = _HeightIntensity * (1 - cameraDistance / 100.0) * _WaveAmplitude.x;
	float2 texCoord 	   = worldPos.xz * 0.05 *_TextureTiling;
	
	// cap or we can go negative
	if (heightIntensity > 0.02)
	{
		AdjustWavesValues(noise, _WavesNoise, _WavesIntensity);

		float2 texCoords[4] = { texCoord * 1.6 + timer * 0.064 + _WavesNoise.x,
								texCoord * 0.8 + timer * 0.032 + _WavesNoise.y,
								texCoord * 0.5 + timer * 0.016 + _WavesNoise.z,
								texCoord * 0.3 + timer * 0.008 + _WavesNoise.w };
		
		float height = 0;
		for (int i = 0; i < 4; ++i)
		{
			height += (heightmap(texCoords[i], _HeightMapScale, _HeightMapScroll) * _WavesIntensity[i]);
		}
		
		worldPos.y += (height * heightIntensity);
	}

  //pixel.position = mul( float4( worldPos.xzy, 1.0f ), worldMatrix );
  //pixel.position = mul( pixel.position, viewProjectionMatrix );
	pixel.position = mul( float4( worldPos.xzy, 1.0f ), worldViewProjectionMatrix );

	pixel.tangent   = tangent;
	pixel.bitangent = cross(normal, tangent);

	pixel.uv.xy 	= worldPos.xz * 0.05 * _TextureTiling;
	pixel.uv.z 		= timer;

	pixel.color 	= vertex.color;
	pixel.normal 	= normal;
	pixel.wind.xy 	= windDir;

	pixel.camPos.xyz = _WorldSpaceCameraPos.xzy;

	pixel.camPosInvDist.xyz = inverseViewMatrix[3].xyz - mul( float4( worldPos.xzy, 1.0f ), worldMatrix );
	pixel.camPosInvDist.w    = distance(worldPos.xzy, inverseWorldMatrix[3]);
	
	pixel.worldPos = worldPos;
	pixel.projPos = Transform_ClipSpacePosToTexCoords(pixel.position);

	return pixel;
}
