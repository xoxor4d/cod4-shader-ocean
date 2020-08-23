
//
// normal calculations
//

float3 UnpackNormal(float4 n) 
{
    n.xyz = n.xyz * 2.0 - 1.0;
    return n.xyz;
}

float3 UnpackNormalRecZ(float4 packednormal) 
{
	float3 normal;
    normal.xy = packednormal.wy * 2 - 1;
    normal.z = sqrt(1 - normal.x*normal.x - normal.y * normal.y);
    return normal;
}

// Project the surface gradient (dhdx, dhdy) onto the surface (n, dpdx, dpdy).
float3  ComputeSurfaceGradient(float3 n, float3 dpdx, float3 dpdy, float dhdx, float dhdy)
{
	float3 r1 = cross(dpdy, n);
	float3 r2 = cross(n, dpdx);

	return (r1 * dhdx + r2 * dhdy) / dot(dpdx, r1);
}

// Move the normal away from the surface normal in the opposite surface gradient direction.
float3 PerturbNormal(float3 n, float3 dpdx, float3 dpdy, float dhdx, float dhdy)
{
	return normalize(n - ComputeSurfaceGradient(n, dpdx, dpdy, dhdx, dhdy));
}

// Returns the surface normal using screen-space partial derivatives of the height field.
float3 ComputeSurfaceNormal(float3 position, float3 normal, float height)
{
	float3 dpdx = ddx(position);
	float3 dpdy = ddy(position);

	float dhdx = ddx(height);
	float dhdy = ddy(height);

	return PerturbNormal(normal, dpdx, dpdy, dhdx, dhdy);
}

float ApplyChainRule(float dhdu, float dhdv, float dud_, float dvd_)
{
	return dhdu * dud_ + dhdv * dvd_;
}

// Calculate the surface normal using the uv-space gradient (dhdu, dhdv)
// Requires height field gradient, double storage.
float3 CalculateSurfaceNormal(float3 position, float3 normal, float2 gradient, float2 duvdx, float2 duvdy)
{
	float3 dpdx = ddx(position);
	float3 dpdy = ddy(position);

	float dhdx = ApplyChainRule(gradient.x, gradient.y, duvdx.x, duvdx.y);
	float dhdy = ApplyChainRule(gradient.x, gradient.y, duvdy.x, duvdy.y);

	return PerturbNormal(normal, dpdx, dpdy, dhdx, dhdy);
}

// Returns the surface normal using screen-space partial derivatives of world position.
// Will result in hard shading normals.
float3 ComputeSurfaceNormal(float3 position)
{
	return normalize(cross(ddx(position), ddy(position)));
}

// portability reasons
float3 mul2x3(float2 val, float3 row1, float3 row2)
{
	float3 res;
	for (int i = 0; i < 3; i++)
	{
		float2 col = float2(row1[i], row2[i]);
		res[i] = dot(val, col);
	}

	return res;
}

float3 ComputeSurfaceNormal(float3 normal, float3 tangent, float3 bitangent, sampler2D tex, float2 uv)
{
	float3x3 tangentFrame = float3x3(normalize(bitangent), normalize(tangent), normal);

#ifndef USE_FILTERING
	normal = UnpackNormal(tex2D(tex, uv));
#else
	float2 duv1 = ddx(uv) * 2;
	float2 duv2 = ddy(uv) * 2;
	normal = UnpackNormal(tex2Dgrad(tex, uv, duv1, duv2));
#endif
	return normalize(mul(normal, tangentFrame));
}

float3x3 ComputeTangentFrame(float3 normal, float3 position, float2 uv)
{
	float3 dp1 = ddx(position);
	float3 dp2 = ddy(position);
	float2 duv1 = ddx(uv);
	float2 duv2 = ddy(uv);

	float3x3 M = float3x3(dp1, dp2, cross(dp1, dp2));
	float3 inverseM1 = float3(cross(M[1], M[2]));
	float3 inverseM2 = float3(cross(M[2], M[0]));
	float3 T = mul2x3(float2(duv1.x, duv2.x), inverseM1, inverseM2);
	float3 B = mul2x3(float2(duv1.y, duv2.y), inverseM1, inverseM2);

	return float3x3(normalize(T), normalize(B), normal);
}

// Returns the surface normal using screen-space partial derivatives of the uv and position coordinates.
float3 ComputeSurfaceNormal(float3 normal, float3 position, sampler2D tex, float2 uv)
{
	float3x3 tangentFrame = ComputeTangentFrame(normal, position, uv);

#ifndef USE_FILTERING
	normal = UnpackNormal(tex2D(tex, uv));
#else
	float2 duv1 = ddx(uv) * 2;
	float2 duv2 = ddy(uv) * 2;
	normal = UnpackNormal(tex2Dgrad(tex, uv, duv1, duv2));
#endif
	return normalize(mul(normal, tangentFrame));
}

float3 ComputeNormal(float4 heights, float strength)
{
	float hL = heights.x;
	float hR = heights.y;
	float hD = heights.z;
	float hT = heights.w;

	float3 normal = float3(hL - hR, strength, hD - hT);
	return normalize(normal);
}

float3 ComputeNormal(sampler2D tex, float2 uv, float texelSize, float strength) // ---------------------------------------------------------------------------------------------- hm was missing .x .y .z .w
{
	float3 off = float3(texelSize, texelSize, 0.0);
	float4 heights;
	heights.x = tex2D(tex, uv.xy - off.xz).x; // hL
	heights.y = tex2D(tex, uv.xy + off.xz).y; // hR
	heights.z = tex2D(tex, uv.xy - off.zy).z; // hD
	heights.w = tex2D(tex, uv.xy + off.zy).w; // hT

	return ComputeNormal(heights, strength);
}




//
// depth refraction
//

// waterTransparency - x = , y = water visibility along eye vector, 
// waterDepthValues - x = water depth in world space, y = view/accumulated water depth in world space
half3 DepthRefraction(float2 waterTransparency, float2 waterDepthValues, float shoreRange, float3 horizontalExtinction,
	half3 refractionColor, half3 shoreColor, half3 surfaceColor, half3 depthColor)
{
	float waterClarity = waterTransparency.x;
	float visibility = waterTransparency.y;
	float waterDepth = waterDepthValues.x;
	float viewWaterDepth = waterDepthValues.y;

	float accDepth = viewWaterDepth * waterClarity; // accumulated water depth
	float accDepthExp = saturate(accDepth / (2.5 * visibility));
	accDepthExp *= (1.0 - accDepthExp) * accDepthExp * accDepthExp + 1; // out cubic

	surfaceColor = lerp(shoreColor, surfaceColor, saturate(waterDepth / shoreRange));
	half3 waterColor = lerp(surfaceColor, depthColor, saturate(waterDepth / horizontalExtinction));

	refractionColor = lerp(refractionColor, surfaceColor * waterColor, saturate(accDepth / visibility));
	refractionColor = lerp(refractionColor, depthColor, accDepthExp);
	refractionColor = lerp(refractionColor, depthColor * waterColor, saturate(waterDepth / horizontalExtinction));
	return refractionColor;
}



//
// snoise
//

float3 mod289(float3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
float2 mod289(float2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
float3 permute(float3 x) { return mod289(((x*34.0) + 1.0)*x); }

float snoise(float2 v)
{
	// Precompute values for skewed triangular grid
	const float4 C = float4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
		0.366025403784439,	// 0.5*(sqrt(3.0)-1.0)
		-0.577350269189626, // -1.0 + 2.0 * C.x
		0.024390243902439	// 1.0 / 41.0
		);


	// First corner (x0)
	float2 i = floor(v + dot(v, C.yy));
	float2 x0 = v - i + dot(i, C.xx);

	// Other two corners (x1, x2)
	float2 i1;
	i1 = (x0.x > x0.y) ? float2(1.0, 0.0) : float2(0.0, 1.0);
	float2 x1 = x0.xy + C.xx - i1;
	float2 x2 = x0.xy + C.zz;

	// Do some permutations to avoid
	// truncation effects in permutation
	i = mod289(i);
	float3 p = permute(
		permute(i.y + float3(0.0, i1.y, 1.0))
		+ i.x + float3(0.0, i1.x, 1.0));

	float3 m = max(0.5 - float3(
		dot(x0, x0),
		dot(x1, x1),
		dot(x2, x2)
		), 0.0);

	m = m*m;
	m = m*m;

	// Gradients: 
	//  41 pts uniformly over a line, mapped onto a diamond
	//  The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

	float3 x = 2.0 * frac(p * C.www) - 1.0;
	float3 h = abs(x) - 0.5;
	float3 ox = floor(x + 0.5);
	float3 a0 = x - ox;

	// Normalise gradients implicitly by scaling m
	// Approximation of: m *= inversesqrt(a0*a0 + h*h);
	m *= 1.79284291400159 - 0.85373472095314 *(a0*a0 + h*h);

	// Compute final noise value at P
	float3 g;
	g.x = a0.x  * x0.x + h.x  * x0.y;
	g.yz = a0.yz * float2(x1.x, x2.x) + h.yz * float2(x1.y, x2.y);
	return 130.0 * dot(m, g);
}



//
// waves
//

float3 GerstnerWaveValues(float2 position, float2 D, float amplitude, float wavelength, float Q, float timer)
{
	float w = 2 * 3.14159265 / wavelength;
	float dotD = dot(position, D);
	float v = w * dotD + timer;
	return float3(cos(v), sin(v), w);
}

half3 GerstnerWaveNormal(float2 D, float A, float Q, float3 vals)
{
	half C = vals.x;
	half S = vals.y;
	half w = vals.z;
	half WA = w * A;
	half WAC = WA * C;
	half3 normal = half3(-D.x * WAC, 1.0 - Q * WA * S, -D.y * WAC);
	return normalize(normal);
}

half3 GerstnerWaveTangent(float2 D, float A, float Q, float3 vals)
{
	half C = vals.x;
	half S = vals.y;
	half w = vals.z;
	half WA = w * A;
	half WAS = WA * S;
	half3 normal = half3(Q * -D.x * D.y * WAS, D.y * WA * C, 1.0 - Q * D.y * D.y * WAS);
	return normalize(normal);
}

float3 GerstnerWaveDelta(float2 D, float A, float Q, float3 vals)
{
	float C = vals.x;
	float S = vals.y;
	float QAC = Q * A * C;
	return float3(QAC * D.x, A * S, QAC * D.y);
}

void GerstnerWave(float2 windDir, float tiling, float amplitude, float wavelength, float Q, float timer, inout float3 position, out half3 normal)
{
	float2 D = windDir;
	float3 vals = GerstnerWaveValues(position.xz * tiling, D, amplitude, wavelength, Q, timer);
	normal = GerstnerWaveNormal(D, amplitude, Q, vals);
	position += GerstnerWaveDelta(D, amplitude, Q, vals);
}

float3 SineWaveValues(float2 position, float2 D, float amplitude, float wavelength, float timer)
{
	float w = 2 * 3.14159265 / wavelength;
	float dotD = dot(position, D);
	float v = w * dotD + timer;
	return float3(cos(v), sin(v), w);
}

half3 SineWaveNormal(float2 D, float A, float3 vals)
{
	half C = vals.x;
	half w = vals.z;
	half WA = w * A;
	half WAC = WA * C;
	half3 normal = half3(-D.x * WAC, 1.0, -D.y * WAC);
	return normalize(normal);
}

half3 SineWaveTangent(float2 D, float A, float3 vals)
{
	half C = vals.x;
	half w = vals.z;
	half WAC = w * A * C;
	half3 normal = half3(0.0, D.y * WAC, 1.0);
	return normalize(normal);
}

float SineWaveDelta(float A, float3 vals)
{
	return vals.y * A;
}

void SineWave(float2 windDir, float tiling, float amplitude, float wavelength, float timer, inout float3 position, out half3 normal)
{
	float2 D = windDir;
	float3 vals = SineWaveValues(position.xz * tiling, D, amplitude, wavelength, timer);
	normal = SineWaveNormal(D, amplitude, vals);
	position.y += SineWaveDelta(amplitude, vals);
}



//
// displacement
//

float2 GetNoise(in float2 position, in float2 timedWindDir)
{
	float2 noise;
	noise.x = snoise(position * 0.015 + timedWindDir * 0.0005); // large and slower noise 
	noise.y = snoise(position * 0.1 + timedWindDir * 0.002); // smaller and faster noise
	return saturate(noise);
}

void AdjustWavesValues(in float2 noise, inout half4 wavesNoise, inout half4 wavesIntensity)
{
	wavesNoise = wavesNoise * half4(noise.y * 0.25, noise.y * 0.25, noise.x + noise.y, noise.y);
	wavesIntensity = wavesIntensity + half4(saturate(noise.y - noise.x), noise.x, noise.y, noise.x + noise.y);
	wavesIntensity = clamp(wavesIntensity, 0.01, 10);
}

// uv in texture space, normal in world space
half3 ComputeNormal(sampler2D normalTexture, float2 worldPos, float2 texCoord, half3 normal, half3 tangent, half3 bitangent, half4 wavesNoise, half4 wavesIntensity, float2 timedWindDir)
{
	float2 noise = GetNoise(worldPos, timedWindDir * 0.5);

	// the fuck
	//AdjustWavesValues(noise, wavesNoise, wavesIntensity);

	float2 texCoords[4] = { texCoord * 1.6 + timedWindDir * 0.064 + wavesNoise.x,
							texCoord * 0.8 + timedWindDir * 0.032 + wavesNoise.y,
							texCoord * 0.5 + timedWindDir * 0.016 + wavesNoise.z,
							texCoord * 0.3 + timedWindDir * 0.008 + wavesNoise.w };

	half3 wavesNormal = half3(0, 1, 0);
//#ifdef USE_DISPLACEMENT
	normal = normalize(normal);
	tangent = normalize(tangent);
	bitangent = normalize(bitangent);
	for (int i = 0; i < 4; ++i)
	{
		wavesNormal += ComputeSurfaceNormal(normal, tangent, bitangent, normalTexture, texCoords[i]) * wavesIntensity[i];
	}

	//wavesNormal = wavesNormal * 4;
//#else
	/* for (int i = 0; i < 4; ++i)
	{
		wavesNormal += UnpackNormal(tex2D(normalTexture, texCoords[i])) * wavesIntensity[i];
	}
	wavesNormal.xyz = wavesNormal.xzy; // flip zy to avoid btn multiplication */
//#endif // #ifdef USE_DISPLACEMENT

	return wavesNormal;
}

#ifdef USE_DISPLACEMENT
float ComputeNoiseHeight(sampler2D heightTexture, float4 wavesIntensity, float4 wavesNoise, float2 texCoord, float2 noise, float2 timedWindDir)
{
	AdjustWavesValues(noise, wavesNoise, wavesIntensity);

	float2 texCoords[4] = { texCoord * 1.6 + timedWindDir * 0.064 + wavesNoise.x,
							texCoord * 0.8 + timedWindDir * 0.032 + wavesNoise.y,
							texCoord * 0.5 + timedWindDir * 0.016 + wavesNoise.z,
							texCoord * 0.3 + timedWindDir * 0.008 + wavesNoise.w };
	float height = 0;
	for (int i = 0; i < 4; ++i)
	{
		height += tex2Dlod(heightTexture, float4(texCoords[i], 0, 0)).w * wavesIntensity[i]; // was .x
	}

	return height;
}

float3 ComputeDisplacement(float3 worldPos, float cameraDistance, float2 noise, float timer, float4 waveSettings, float4 waveAmplitudes, float4 wavesIntensity, float4 waveNoise,
							out half3 normal, out half3 tangent)
{
	float2 windDir = waveSettings.xy;
	float waveSteepness = waveSettings.z;
	float waveTiling = waveSettings.w;

	//TODO: improve motion/simulation instead of just noise
	//TODO: fix UV due to wave distortion

	wavesIntensity = normalize(wavesIntensity);
	waveNoise = half4(noise.x - noise.x * 0.2 + noise.y * 0.1, noise.x + noise.y * 0.5 - noise.y * 0.1, noise.x, noise.x) * waveNoise;
	half4 wavelengths = half4(1, 4, 3, 6) + waveNoise;
	half4 amplitudes = waveAmplitudes + half4(0.5, 1, 4, 1.5) * waveNoise;

	// reduce wave intensity base on distance to reduce aliasing
	wavesIntensity *= 1.0 - saturate(half4(cameraDistance / 120.0, cameraDistance / 150.0, cameraDistance / 170.0, cameraDistance / 400.0));

	// compute position and normal from several sine and gerstner waves
	tangent = normal = half3(0, 1, 0);
	float2 timers = float2(timer * 0.5, timer * 0.25);
	for (int i = 2; i < 4; ++i)
	{
		float A = wavesIntensity[i] * amplitudes[i];
		float3 vals = SineWaveValues(worldPos.xz * waveTiling, windDir, A, wavelengths[i], timer);
		normal += wavesIntensity[i] * SineWaveNormal(windDir, A, vals);
		tangent += wavesIntensity[i] * SineWaveTangent(windDir, A, vals);
		worldPos.y += SineWaveDelta(A, vals);
	}

	// using normalized wave steepness, tranform to Q
	float2 Q = waveSteepness / ((2 * 3.14159265 / wavelengths.xy) * amplitudes.xy);
	for (int j = 0; j < 2; ++j)
	{
		float A = wavesIntensity[j] * amplitudes[j];
		float3 vals = GerstnerWaveValues(worldPos.xz * waveTiling, windDir, A, wavelengths[j], Q[j], timer);
		normal += wavesIntensity[j] * GerstnerWaveNormal(windDir, A, Q[j], vals);
		tangent += wavesIntensity[j] * GerstnerWaveTangent(windDir, A, Q[j], vals);
		worldPos += GerstnerWaveDelta(windDir, A, Q[j], vals);
	}

	normal = normalize(normal);
	tangent = normalize(tangent);
	if (length(wavesIntensity) < 0.01)
	{
		normal = half3(0, 1, 0);
		tangent = half3(0, 0, 1);
	}

	return worldPos;
}
#endif


//
// fresnel
//

half FresnelValue(float2 refractionValues, float3 normal, float3 eyeVec)
{
	// R0 is a constant related to the index of refraction (IOR).
	float R0 = refractionValues.x;
	// This value modifies current fresnel term. If you want to weaken
	// reflections use bigger value.
	float refractionStrength = refractionValues.y;
#ifdef SIMPLIFIED_FRESNEL
	return R0 + (1.0f - R0) * pow(1.0f - dot(eyeVec, normal), 5.0f);
#else		
	float angle = 1.0f - saturate(dot(normal, eyeVec));
	float fresnel = angle * angle;
	fresnel *= fresnel;
	fresnel *= angle;
	return saturate(fresnel * (1.0f - saturate(R0)) + R0 - refractionStrength);
#endif // #ifdef SIMPLIFIED_FRESNEL
}

// lightDir, eyeDir and normal in world space
half3 ReflectedRadiance(float shininess, half3 specularValues, half3 lightColor, float3 lightDir, float3 eyeDir, float3 normal, float fresnel)
{
	float shininessExp = specularValues.z;

#ifdef BLINN_PHONG
	// a variant of the blinn phong shading
	float specularIntensity = specularValues.x * 0.0075;

	float3 H = normalize(eyeDir + lightDir);
	float e = shininess * shininessExp * 800;
	float kS = saturate(dot(normal, lightDir));
	half3 specular = kS * specularIntensity * pow(saturate(dot(normal, H)), e) * sqrt((e + 1) / 2);
	specular *= lightColor;
#else
	float2 specularIntensity = specularValues.xy;
	// reflect the eye vector such that the incident and emergent angles are equal
	float3 mirrorEye = reflect(-eyeDir, normal);
	half dotSpec = saturate(dot(mirrorEye, lightDir) * 0.5f + 0.5f);
	half3 specular = (1.0f - fresnel) * saturate(lightDir.y) * pow(dotSpec, specularIntensity.y) * (shininess * shininessExp + 0.2f) * lightColor;
	specular += specular * specularIntensity.x * saturate(shininess - 0.05f) * lightColor;
#endif // #ifdef BLINN_PHONG
	return specular;
}


//
// mean sky radiance
// 

#ifdef USE_MEAN_SKY_RADIANCE

// V, N, Tx, Ty in world space
float2 U(float2 zeta, float3 V, float3 N, float3 Tx, float3 Ty)
{
	float3 f = normalize(float3(-zeta, 1.0)); // tangent space
	float3 F = f.x * Tx + f.y * Ty + f.z * N; // world space
	float3 R = 2.0 * dot(F, V) * F - V;
	return  dot(F, V);
}

// viewDir and normal in world space
half3 MeanSkyRadiance(samplerCUBE skyTexture, float3 viewDir, half3 normal)
{
	if (dot(viewDir, normal) < 0.0)
	{
		normal = reflect(normal, viewDir);
	}
	float3 ty = normalize(float3(0.0, normal.z, -normal.y));
	float3 tx = cross(ty, normal);

	const float eps = 0.001;
	float2 u0 = U(float2(0, 0), viewDir, normal, tx, ty) * 0.05;
	float2 dux = 2.0 * (float2(eps, 0.0) - u0) / eps;
	float2 duy = 2.0 * (float2(0, eps) - u0) / eps;
	return texCUBE(skyTexture, float3(u0.xy, 1.0)).rgb; //TODO: transform hemispherical cordinates to cube or use a 2d texture
}

#endif // #ifdef USE_MEAN_SKY_RADIANCE


//
// bicubic filtering
//

float4 cubic(float v)
{
	float4 n = float4(1.0, 2.0, 3.0, 4.0) - v;
	float4 s = n * n * n;
	float x = s.x;
	float y = s.y - 4.0 * s.x;
	float z = s.z - 4.0 * s.y + 6.0 * s.x;
	float w = 6.0 - x - y - z;
	return float4(x, y, z, w) * (1.0 / 6.0);
}

// 4 taps bicubic filtering, requires sampler to use bilinear filtering
float4 tex2DBicubic(sampler2D tex, float texSize, float2 texCoords)
{
	float2 texSize2 = texSize;
	float2 invTexSize = 1.0 / texSize2;

	texCoords = texCoords * texSize2 - 0.5;
	float2 fxy = frac(texCoords);
	texCoords -= fxy;

	float4 xcubic = cubic(fxy.x);
	float4 ycubic = cubic(fxy.y);

	float4 c = texCoords.xxyy + float2(-0.5, +1.5).xyxy;

	float4 s = float4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
	float4 offset = c + float4(xcubic.yw, ycubic.yw) / s;

	offset *= invTexSize.xxyy;

	float4 sample0 = tex2D(tex, offset.xz);
	float4 sample1 = tex2D(tex, offset.yz);
	float4 sample2 = tex2D(tex, offset.xw);
	float4 sample3 = tex2D(tex, offset.yw);

	float sx = s.x / (s.x + s.y);
	float sy = s.z / (s.z + s.w);

	return lerp(lerp(sample3, sample2, sx), lerp(sample1, sample0, sx), sy);
}



//
// foam
//

half FoamColor(sampler2D tex, float2 texCoord, float2 texCoord2, float2 ranges, half2 factors, float waterDepth, half baseColor)
{
	float f1 = tex2D(tex, texCoord).r;
	float f2 = tex2D(tex, texCoord2).r;
	return lerp(f1 * factors.x + f2 * factors.y, baseColor, smoothstep(ranges.x, ranges.y, waterDepth));
}

// surfacePosition, depthPosition, eyeVec in world space
// waterDepth is the horizontal water depth in world space
half FoamValue(sampler2D shoreTexture, sampler2D foamTexture, float2 foamTiling,
	float4 foamNoise, float2 foamSpeed, float3 foamRanges, float maxAmplitude,
	float3 surfacePosition, float3 depthPosition, float3 eyeVec, float waterDepth,
	float2 timedWindDir, float timer)
{
	float2 position = (surfacePosition.xz + eyeVec.xz * 0.1) * 0.5;

	float s = sin(timer * 0.01 + depthPosition.x);
	float2 texCoord = position + timer * 0.01 * foamSpeed + s * 0.05;
	s = sin(timer * 0.01 + depthPosition.z);
	float2 texCoord2 = (position + timer * 0.015 * foamSpeed + s * 0.05) * -0.5; // also flip
	float2 texCoord3 = texCoord * foamTiling.x;
	float2 texCoord4 = (position + timer * 0.015 * -foamSpeed * 0.3 + s * 0.05) * -0.5 * foamTiling.x; // reverse direction
	texCoord *= foamTiling.y;
	texCoord2 *= foamTiling.y;

	float2 ranges = foamRanges.xy;
	ranges.x += snoise(surfacePosition.xz + foamNoise.z * timedWindDir) * foamNoise.x;
	ranges.y += snoise(surfacePosition.xz + foamNoise.w * timedWindDir) * foamNoise.y;
	ranges = clamp(ranges, 0.0, 10.0);

	float foamEdge = max(ranges.x, ranges.y);
	half deepFoam = FoamColor(foamTexture, texCoord, texCoord2, float2(ranges.x, foamEdge), half2(1.0, 0.5), waterDepth, 0.0);
	half foam = 0;//= FoamColor(shoreTexture, texCoord3 * 0.25, texCoord4, float2(0.0, ranges.x), half2(0.75, 1.5), waterDepth, deepFoam);

	// high waves foam
	//if (surfacePosition.y - foamRanges.z > 0.0001f)
	{
		half amount = saturate((surfacePosition.y - foamRanges.z) / maxAmplitude) * 0.25;
		foam += (tex2D(shoreTexture, texCoord3).x + tex2D(shoreTexture, texCoord4).x * 0.5f) * amount;
	}

	return foam;
}

// surfacePosition, depthPosition, eyeVec in world space
// waterDepth is the horizontal water depth in world space
half FoamValueAlphaShore(sampler2D 	shoreTexture, sampler2D foamTexture		, float2 foamTiling,
						 float4 	foamNoise	, float2 	foamSpeed		, float3 foamRanges, 
						 float 		maxAmplitude, float3 	surfacePosition , float3 depthPosition, 
						 float3 	eyeVec		, float 	waterDepth		, float2 timedWindDir, 
						 float 		timer)
{
	float2 position = (surfacePosition.xz + eyeVec.xz * 0.1) * 0.5;

	float s = sin(timer * 0.01 + depthPosition.x);
	float2 texCoord = position + timer * 0.01 * foamSpeed + s * 0.05;
	s = sin(timer * 0.01 + depthPosition.z);
	float2 texCoord2 = (position + timer * 0.015 * foamSpeed + s * 0.05) * -0.5; // also flip
	float2 texCoord3 = texCoord * foamTiling.x;
	float2 texCoord4 = (position + timer * 0.015 * -foamSpeed * 0.3 + s * 0.05) * -0.5 * foamTiling.x; // reverse direction
	
	texCoord *= foamTiling.y;
	texCoord2 *= foamTiling.y;

	float2 ranges = foamRanges.xy;
	ranges.x += snoise(surfacePosition.xz + (foamNoise.z) * timedWindDir) * foamNoise.x;
	ranges.y += snoise(surfacePosition.xz + foamNoise.w * timedWindDir) * foamNoise.y;
	ranges = clamp(ranges, 0.0, 10.0);

	float foamEdge = max(ranges.x, ranges.y);
	half foam = FoamColor(foamTexture, texCoord, texCoord2, float2(ranges.x, foamEdge), half2(1.0, 4.5), waterDepth, 0);//-2.0);
	//half foam = FoamColor(shoreTexture, texCoord3 * 0.25, texCoord4, float2(0.0, ranges.x), half2(0.75, 1.5), 1.2, deepFoam);

	return foam;
}