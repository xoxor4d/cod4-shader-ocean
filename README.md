## A Call of Duty 4 - Realtime / dynamic ocean shader, fully customizable in-game via the use of the IW3xo client

#### Using the shader:
1. Clone the repo or download as zip
2. Copy all files into the cod4 root directory
3. Open the "shader_ocean.gdt" in Asset Manager, compile the material and the xmodel
4. Compile shaders, "worldfx_ocean" and "worldfx_ocean_unlit" by either using https://xoxor4d.github.io/projects/cod4-compileTools/ or shader_tool directly (https://xoxor4d.github.io/tutorials/hlsl-intro/#compiling)
5. Place the "ocean_plane" xmodel into your map or apply the material to any other kind of xmodel/terrain patch
6. Tweak in-game using IW3xo (The shader ships with in-game tweaking enabled, requiring IW3xo to be used, see below to disable this if needed)

#### Tweaking in-game using the GUI:
1. Use IW3xo and the built in command "/devgui" -> ocean tab to tweak the shader however you like. Use the export button to export your settings. (root/iw3xo/shader_settings/ocean_settings.txt)
2. Overwrite the static shader constants inside the __#else block__ (if __USE_CUSTOM_CONSTANTS__ is not defined) in both the vertex and pixelshader. Note that both are sharing 3 of the exported constants.
3. The shader ships with in-game tweaking enabled, requiring IW3xo to be used. This can be disabled by commenting "//" __#define USE_CUSTOM_CONSTANTS__ in both the vertex and pixelshader.
4. Disabling __USE_CUSTOM_CONSTANTS__ will enable vanilla cod4 usage. You have to do this before you ship your mod/map. 

#### Tweaking in-game using an addon fastfile:
- IW3xo supports loading/reloading of fastfiles and its assets in-game
- Expecting that you already have the shader up and running in your map/mod, do the following:
  - Include the following in your fastfile zone:

  > material,dynamic_ocean  
  > techset,mc_worldfx_ocean  
  > techset,wc_worldfx_ocean  
  > techset,worldfx_ocean  

  - Modify the shader and recompile it
  - Build the fastfile
  - Load up your map/mod thats including the ocean shader
  - Use the following command to load your zone file: "/loadzone your_zone_name"
  - The ocean shader should now reflect your changes
  - Rinse and repeat

#### Note:
- Flat plane models with lots of vertices work best (for obvious reasons)
- The shader was written to be used with models. It does however work with terrain patches and alike. It just wont look as good.
- Duplicate shader files and assign them to a new material if you need different settings for different enviroments
- The _unlit variant will be used in radiant or in-game when r_fullbright is turned on. Please note that the _unlit variant does not support custom shader constants, meaning it cannot be controlled in-game. (use to the addon fastfile method instead) 

___

Project Page:  
https://xoxor4d.github.io/projects/cod4-ocean/

IW3xo:  
https://xoxor4d.github.io/projects/iw3xo/

Custom CoD4 Compiletools:  
https://xoxor4d.github.io/projects/cod4-compileTools/

Discord:  
https://discord.gg/t5jRGbj

## Credits
- https://github.com/tuxalin/water-shader
