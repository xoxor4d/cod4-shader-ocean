// * xoxor4d.github.io
// * original shader : https://github.com/tuxalin/water-shader

#define PC
#define IS_VERTEX_SHADER 0
#define IS_PIXEL_SHADER 1

// use custom code constants (iw3xo-devgui)
#define USE_CUSTOM_CONSTANTS

//#define REVERSED_Z // reversed depth
#define USE_DISPLACEMENT
#define USE_FOAM
#define USE_SHORE_FOAM
#define USE_MEAN_SKY_RADIANCE // can be used for any sort of cubemap (eg. a "custom" reflection probe)
//#define USE_REFLECTION_PROBES // uses the ^ cubemap otherwise (only if USE_MEAN_SKY_RADIANCE is defined)

//#define DEBUG_NORMALS
//#define DEBUG_WATERDEPTH
//#define DEBUG_PURE_REFRECTION
//#define DEBUG_SHORE_FADE
//#define DEBUG_POSITION_FROM_DEPTH


#include <shader_vars.h>
#include <lib/transform.hlsl>
#include <lib/utility.hlsl>
#include <lib/ocean_utility.hlsl>

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

struct PixelOutput
{
	float4 color : COLOR;
};

static float2 rcpres = float2(texelSizeX, texelSizeY); // size of 1 pixel horz. and vert.
static float  FOV 	 = 80;
static const float2 g_InvFocalLen = { tan(0.5f * radians(FOV)) / rcpres.y * rcpres.x, tan(0.5f * radians(FOV)) }; 

float3 fetch_eye_pos(float2 uv, float eye_z)
{
    uv = (uv * float2(-2.0, 2.0) - float2(-1.0, 1.0));
#ifdef REVERSED_Z
	eye_z = 1.0f - eye_z;
#endif

    return float3(uv * g_InvFocalLen * eye_z, eye_z);
}

// dx9 : VFACE : Floating-point scalar that indicates a back-facing primitive. A negative value faces backwards, while a positive value faces the camera.
// dx9 : VPOS  : screen coords of current pixel

PixelOutput ps_main( const PixelInput pixel)
{
    PixelOutput fragment;

#ifdef USE_CUSTOM_CONSTANTS

	// both vs / ps
	float4 _WaveAmplitude 	= float4(1 * filterTap[0]);
	float4 _WavesIntensity 	= float4(1 * filterTap[1]);
	float4 _WavesNoise 		= float4(1 * filterTap[2]);

	// colors
	float3 _AmbientColor 	= float3(1 * filterTap[3].xyz);
	float  _AmbientDensity 	= 		 1 * filterTap[3].w;

	float3 _ShoreColor 		= float3(1 * filterTap[4].xyz);
	float  _DiffuseDensity 	= 		 1 * filterTap[4].w;

	float3 _SurfaceColor 	= float3(1 * filterTap[5].xyz);
	float  _NormalIntensity = 		 1 * filterTap[5].w;

	float3 _DepthColor 		= float3(1 * filterTap[6].xyz);
	float  _ShoreFade 		= 		 1 * filterTap[6].w;
	
	// refraction
	float3 _RefractionValues = float3(1 * filterTap[7].xyz);
	float  _RefractionScale  = 		  1 * filterTap[7].w;

	float3 _HorizontalExtinction = float3(1 * colorMatrixR.xyz);
	float  _Distortion 			 = 		  1 * colorMatrixR.w;

	float _WaterClarity 		= 1 * colorMatrixG.x;
	float _WaterTransparency 	= 1 * colorMatrixG.y;
	float _RadianceFactor 		= 1 * colorMatrixG.z;				// 1 float left empty

	// spec
	float3 _SpecularValues 		= float3(1 * colorMatrixB.xyz);
	float  _Shininess 			= 		 1 * colorMatrixB.w;
	
	// shore/foam
	float3 _FoamTiling 			= float3(1 * dofEquationViewModelAndFarBlur.xyz);
	float  _FoamSpeed 			= 		 1 * dofEquationViewModelAndFarBlur.w;

	float3 _FoamRanges 			= float3(1 * dofEquationScene.xyz);
	float  _FoamIntensity 		= 		 1 * dofEquationScene.w;

	float4 _FoamNoise			= float4(1 * dofLerpScale); //  y: inner blend z: inner blend to outer dist //float4(0.1, 0.12, 0.1, 0.2);

#else // ------------------------------------------

	// both vs / ps
	const float4 _WaveAmplitude = float4(0.72, 0.4, 0.46, 1.93);
		  float4 _WavesIntensity = float4(0.28, 1.95, 1, 1.9);
		  float4 _WavesNoise = float4(0.05, 0.45, 1.18, 0.28);

	// colors
	const float3 _AmbientColor = float3(1, 1, 1);
	const float  _AmbientDensity = -0.0075;
	const float3 _ShoreColor = float3(1, 1, 1);
	const float  _DiffuseDensity = -0.2649;
	const float3 _SurfaceColor = float3(0.871, 0.953, 0.804);
	const float  _NormalIntensity = 0.9549;
	const float3 _DepthColor = float3(0.914, 0.988, 1.047);
	const float  _ShoreFade = 0.0073;

	// refraction
	const float3 _RefractionValues = float3(-0.2637, 0.4148, 1);
	const float  _RefractionScale = 0.005;
	const float3 _HorizontalExtinction = float3(100, 250, 500);
	const float  _Distortion = 0.4;
	const float  _WaterClarity = 12;
	const float  _WaterTransparency = 500;
	const float  _RadianceFactor = -1.75;

	// spec
	const float3 _SpecularValues = float3(17.75, 43, 5.5);
	const float  _Shininess = 0.68;

	// shore/foam
	const float3 _FoamTiling = float3(0.075, 0.25, -0.7);
	const float  _FoamSpeed = 150;
	const float3 _FoamRanges = float3(1.8, 2.2, 2.46);
		  float  _FoamIntensity = 14;
	const float4 _FoamNoise = float4(1, 1, 1, 1);

#endif // USE_CUSTOM_CONSTANTS

	// --------------------------------------------------------

	float4 _WorldSpaceCameraPos = float4(pixel.camPos.xyz, 1.0);

	float  timer 		= pixel.uv.z;
	float2 windDir 		= pixel.wind.xy;
	float2 timedWindDir = pixel.wind.xy * timer;
	float2 ndcPos 		= float2(pixel.projPos.xy / pixel.projPos.w);
	float3 eyeDir 		= normalize(_WorldSpaceCameraPos.xzy - float3(pixel.worldPos.x, pixel.worldPos.y + 10000, pixel.worldPos.z)); // hack, "worldPos" is still in local space, so we are working with the model origin resulting in odd behaviour
	float3 eyeDirWorld	= normalize(pixel.camPosInvDist);

	float3 surfacePosition 	= pixel.worldPos;
	half3  lightColor 		= sunDiffuse.rgb;

	// compensate _FoamIntensity by the amount of sunlight so we dont need to tweak the shader everytime we increase/decrease the maps sunlight
	_FoamIntensity = _FoamIntensity / (((sunDiffuse.r + sunDiffuse.g + sunDiffuse.b) * 0.333f) * 1.5f + 0.5);

	//wave normal
	half3 normal = ComputeNormal(normalMapSampler, surfacePosition.xz, pixel.uv, pixel.normal, pixel.tangent, pixel.bitangent, _WavesNoise, _WavesIntensity, timedWindDir);
	normal = normalize(lerp(pixel.normal, normalize(normal), _NormalIntensity));

	//not propper
	float3 depthPosition = fetch_eye_pos(ndcPos, 1);
	depthPosition 	= depthPosition.xzy;
	depthPosition.y = surfacePosition.y - 100; // hack

	//not propper
	float waterDepth, viewWaterDepth, waterDepthDbg;

	waterDepth = tex2D(floatZSampler, ndcPos).r;
	waterDepth -= pixel.projPos.w;
	waterDepth /= 2;
	viewWaterDepth = waterDepth; // meh
	waterDepthDbg  = waterDepth;

	// refraction
	float2 dudv = float2(ndcPos.x, ndcPos.y * 0.98);
	{
		// refraction based on water depth
		float refractionScale = _RefractionScale * min(waterDepth, 1.0f);

		float2 delta = float2(sin(timer + 3.0f * abs(1/waterDepth)),
							  sin(timer + 5.0f * abs(1/waterDepth)));
		
		dudv = dudv + windDir * delta * refractionScale;
	}

	half3 pureRefractionColor = tex2D(colorMapPostSunSampler, dudv).rgb; // half3(1,0,0); // _RefractionTexture

	float2 waterTransparency = float2(_WaterClarity, _WaterTransparency);
	float2 waterDepthValues  = float2(waterDepth, viewWaterDepth);
	float  shoreRange 		 = max(_FoamRanges.x, _FoamRanges.y) * 2.0;
	half3  refractionColor 	 = DepthRefraction(waterTransparency, waterDepthValues, shoreRange, _HorizontalExtinction, pureRefractionColor, _ShoreColor, _SurfaceColor, _DepthColor);

	// compute light's reflected radiance
	float3 lightDir  = normalize(sunPosition.xzy);
	half   fresnel 	 = FresnelValue(_RefractionValues, normal, eyeDir);
	half3  specularColor = ReflectedRadiance(_Shininess, _SpecularValues, lightColor, lightDir, eyeDirWorld, normal, fresnel);


	// -----------------------
	// compute reflected color
	
#ifdef USE_MEAN_SKY_RADIANCE // compute sky's reflected radiance
	half3 reflectColor = fresnel * MeanSkyRadiance(skyMapSampler, eyeDirWorld, normal) * _RadianceFactor;
#else
	half3 reflectColor = 0;
#endif // #ifdef USE_MEAN_SKY_RADIANCE


	dudv = ndcPos + _Distortion * normal.xz;


#ifdef USE_REFLECTION_PROBES // # using reflectionprobes
	
	float3 reflectVec = normalize(reflect(pixel.camPosInvDist.xyz, normal.xzy * _Distortion));
	reflectColor += texCUBE(reflectionProbeSampler, reflectVec);

#else // # user defined cubemap for reflection (a sky in this case)
	#ifdef USE_MEAN_SKY_RADIANCE

		float3 reflectVec = normalize(reflect(eyeDirWorld, normal.xzy * _Distortion)); // normalize(reflect(pixel.camPosInvDist.yxz, normal.xzy * _Distortion));
		
		// rotate reflection vector by 180 deg
		float4x4 rot_mat = rotation_matrix_4x4(float3(0.0, 0.0, M_PI), float4(0.0, 0.0, 0.0, 1.0));
		reflectVec = mul(reflectVec, rot_mat);

		reflectColor += texCUBE(skyMapSampler, reflectVec);

	#endif
#endif // #ifdef USE_REFLECTION_PROBES


	// -----------------------
	// wave foam

	float maxAmplitude = max(max(_WaveAmplitude.x, _WaveAmplitude.y), _WaveAmplitude.z);

#ifdef USE_FOAM

						//_ShoreTexture (rgb) _FoamTexture (rgb) (modified FoamValue function, doesnt use colorMapSampler anymore)
	half foam = FoamValue(specularMapSampler, colorMapSampler, _FoamTiling, _FoamNoise, _FoamSpeed * windDir, _FoamRanges, maxAmplitude, surfacePosition, depthPosition, eyeDir, waterDepth, timedWindDir, timer);
 		 foam = foam * _FoamIntensity;

#else
	half foam = 0;
#endif // #ifdef USE_FOAM


	// ----------------------

	half  shoreFade = saturate(waterDepth * _ShoreFade);
	
	// ambient + diffuse
	half3 ambientColor = _AmbientColor.rgb * _AmbientDensity + saturate(dot(normal, lightDir)) * _DiffuseDensity;
	
	// refraction color with depth based color
	pureRefractionColor = lerp(pureRefractionColor, reflectColor, fresnel * saturate(waterDepth / (_FoamRanges.x * 0.4)));
	pureRefractionColor = lerp(pureRefractionColor, _ShoreColor, 0.30 * shoreFade);
	
	// compute final color
	half3 color = lerp(refractionColor, reflectColor, fresnel);
	color = saturate(ambientColor + color + max(specularColor, foam * lightColor));
	color = lerp(pureRefractionColor + specularColor * shoreFade, color, shoreFade);


	// -----------------------
	// shore foam

#ifdef USE_SHORE_FOAM

	timer *= 0.25; // reduce shore foam scroll
	half alphaFoam = FoamValueAlphaShore(specularMapSampler, specularMapSampler, _FoamTiling, _FoamNoise, _FoamSpeed * windDir, _FoamRanges, maxAmplitude, surfacePosition, depthPosition, eyeDirWorld, waterDepth * 0.25, timedWindDir, timer);
	
	float dist_cam_to_vert = pixel.camPosInvDist.w;
	float sprinkle_scalar  = clamp(toRange2(0, 450, 1, 0, dist_cam_to_vert), 0, 1);

	// fade out shore foam
	if(dist_cam_to_vert >= 0 && dist_cam_to_vert < 450)
	{
		alphaFoam = (alphaFoam * sprinkle_scalar) * _FoamIntensity * 0.0075;
		color += (alphaFoam);
	}

#endif // #ifdef USE_SHORE_FOAM


	// -----------------------
	// final color

	fragment.color.rgb = color.rgb;

	// fade out water towards shore depending on waterDepth
	waterDepth = clamp(waterDepth * 0.15, 0, 1);
	fragment.color.a = pixel.color.a * (waterDepth);
	

	// -----------------------
	// debug stuff

#ifdef DEBUG_NORMALS
	color.rgb = 0.5 + 2 * ambientColor + specularColor + clamp(dot(normal, lightDir), 0, 1) * 0.5;		
	fragment.color.rgb = color.rgb + (fragment.color.rgb * 0.0001);
#endif

#ifdef DEBUG_WATERDEPTH
	color.rgb = float3(waterDepthDbg / 100, 0, 0); 
	fragment.color.rgb = color.rgb + (fragment.color.rgb * 0.0001);
#endif

#ifdef DEBUG_PURE_REFRECTION
	color.rgb = float3(pureRefractionColor);  	
	fragment.color.rgb = color.rgb + (fragment.color.rgb * 0.0001);
#endif

#ifdef DEBUG_POSITION_FROM_DEPTH
	color.rgb = float3(depthPosition.xzy);  	
	fragment.color.rgb = color.rgb + (fragment.color.rgb * 0.0001);
#endif

#ifdef DEBUG_SHORE_FADE
	color.rgb = float3(shoreFade, shoreFade, shoreFade);  	
	fragment.color.rgb = color.rgb + (fragment.color.rgb * 0.0001);
#endif

	return fragment;
}
