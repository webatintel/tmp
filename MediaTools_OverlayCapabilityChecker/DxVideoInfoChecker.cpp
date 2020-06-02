#include "DxVideoInfoChecker.h"
#include <iostream>
#include <iomanip>
#include <string>

#include "dxgi_enums_string.h"


std::string WStringToString(const std::wstring& wstr)
{
	std::string str(wstr.length(), ' ');
	std::copy(wstr.begin(), wstr.end(), str.begin());
	return str;

}

void DxVideoInfoChecker::Initialize() {
	
	
	
}

void DxVideoInfoChecker::CheckOverlayCapability() {
	
	Microsoft::WRL::ComPtr<IDXGIFactory4> dxgi_factory;
	if(FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&dxgi_factory))))
		std::cout << "Create DXGI factory error." << std::endl;

	Microsoft::WRL::ComPtr<IDXGIAdapter> dxgi_adapter;
	if(FAILED(dxgi_factory->EnumAdapters(0,&dxgi_adapter)))
		std::cout << "Enum adapters error." << std::endl;

	DXGI_ADAPTER_DESC adapter_desc;
	dxgi_adapter->GetDesc(&adapter_desc);
	std::wstring wadapter_description(adapter_desc.Description);
	std::string adapter_description = WStringToString(wadapter_description);
	std::cout << "Default graphics card: " << adapter_description << std::endl;
	
	UINT flags = 0;
	D3D_FEATURE_LEVEL feature_levels[] = {D3D_FEATURE_LEVEL_11_1};
	D3D_FEATURE_LEVEL feature_level_out = D3D_FEATURE_LEVEL_11_1;
	Microsoft::WRL::ComPtr<ID3D11Device> d3d11_device;
	Microsoft::WRL::ComPtr<ID3D11DeviceContext> d3d11_device_context;
	HRESULT hr = D3D11CreateDevice(dxgi_adapter.Get(),   //IDXGIAdapter
		D3D_DRIVER_TYPE_UNKNOWN,    //DriverType
		0,					    	//Software
		flags,						//Flags
		feature_levels,			    //*pFeatureLevels
		1,							//FeatureLevels
		D3D11_SDK_VERSION,			//SDKVersion
		&d3d11_device,				//**ppDevice
		&feature_level_out,			//*pFeatureLevel
		&d3d11_device_context       //**ppImmediateContext
	);
	if (FAILED(hr)) {
		std::cout<<"Create D3D11 device failed. Code:"<<std::hex<<hr<<std::endl;
		return;
	}
	
	LARGE_INTEGER ldriver_version;
	if (SUCCEEDED(dxgi_adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &ldriver_version))) {
		char driver_version[100];
		sprintf_s(driver_version, 100, "%d.%d.%d.%d", HIWORD(ldriver_version.HighPart),
			LOWORD(ldriver_version.HighPart), HIWORD(ldriver_version.LowPart),
			LOWORD(ldriver_version.LowPart));
		std::cout<<"Driver version:"<<driver_version<<std::endl;
	}

	UINT i = 0;
	while (true) {
		Microsoft::WRL::ComPtr<IDXGIOutput> output;
		if (FAILED(dxgi_adapter->EnumOutputs(i++, &output)))
			break;
		std::cout<<"    -------Display "<<i<<"-------"<<std::endl;

		DXGI_OUTPUT_DESC desc;
		if (SUCCEEDED(output->GetDesc(&desc))) {
			std::wstring wdevice_name(desc.DeviceName);
			//UINT width = desc.DesktopCoordinates.right - desc.DesktopCoordinates.left;
			//UINT height = desc.DesktopCoordinates.bottom - desc.DesktopCoordinates.top;
			std::cout<<"DeviceName:"<<WStringToString(wdevice_name)<<std::endl;
			//std::cout << "Resulotion:" << width << "x" << height<<std::endl;
		}

		Microsoft::WRL::ComPtr<IDXGIOutput3> output3;
		if (FAILED(output.As(&output3)))
			continue;
		
		std::cout << "Supported overlay format:\t\t(IDXGIOutput3::CheckOverlaySupport)"<<std::endl;
		UINT overlay_support_flags;
		for (UINT i = 0; i < DXGI_FORMAT_MAX; i++) {
			output3->CheckOverlaySupport(static_cast<DXGI_FORMAT>(i), d3d11_device.Get(),
				&overlay_support_flags);
			if (overlay_support_flags) {
				std::cout << "	" << DXGI_FORMAT_STRING[i] 
					<<"\tDIRECT:"<<bool(DXGI_OVERLAY_SUPPORT_FLAG_DIRECT|overlay_support_flags) 
					<<"\tSCALING:"<<bool(DXGI_OVERLAY_SUPPORT_FLAG_SCALING|overlay_support_flags)
					<< std::endl;
			}
		}

		Microsoft::WRL::ComPtr<IDXGIOutput4> output4;
		if (FAILED(output.As(&output4)))
			continue;
		std::cout << "Supported overlay format-colorspace:\t (IDXGIOutput4::CheckOverlayColorSpaceSupport)" << std::endl;
		UINT colorspace_support_flag;
		for (UINT format_idx = 0; format_idx < DXGI_FORMAT_MAX; format_idx++) {
			for (UINT colorspace_idx = 0; colorspace_idx < DXGI_COLOR_SPACE_TYPE_MAX; colorspace_idx++) {

				output4->CheckOverlayColorSpaceSupport(static_cast<DXGI_FORMAT>(format_idx),
					static_cast<DXGI_COLOR_SPACE_TYPE>(colorspace_idx),
					d3d11_device.Get(),
					&colorspace_support_flag);

				if (colorspace_support_flag) {
					std::cout << "	" <<std::left<<std::setw(35)<< DXGI_FORMAT_STRING[format_idx] << "\t" <<
						DXGI_COLOR_SPACE_TYPE_STRING[colorspace_idx] << std::endl;
				}

			}
		}
		
	}
}

void DxVideoInfoChecker::CheckVideoProcessorCapability() {
	std::cout << "Check video processor capability. Output format supported at 1080p 60fps:" << std::endl;
	Microsoft::WRL::ComPtr<IDXGIFactory4> dxgi_factory;
	if(FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&dxgi_factory))))
		std::cout << "Create DXGI factory error." << std::endl;

	Microsoft::WRL::ComPtr<IDXGIAdapter> dxgi_adapter;
	if(FAILED(dxgi_factory->EnumAdapters(0,&dxgi_adapter)))
		std::cout << "Enum adapters error." << std::endl;

	UINT flags = 0;
	D3D_FEATURE_LEVEL feature_levels[] = {D3D_FEATURE_LEVEL_11_1};
	D3D_FEATURE_LEVEL feature_level_out = D3D_FEATURE_LEVEL_11_1;
	Microsoft::WRL::ComPtr<ID3D11Device> d3d11_device;
	Microsoft::WRL::ComPtr<ID3D11DeviceContext> d3d11_device_context;
	HRESULT hr = D3D11CreateDevice(dxgi_adapter.Get(),   //IDXGIAdapter
		D3D_DRIVER_TYPE_UNKNOWN,    //DriverType
		0,					    	//Software
		flags,						//Flags
		feature_levels,			    //*pFeatureLevels
		1,							//FeatureLevels
		D3D11_SDK_VERSION,			//SDKVersion
		&d3d11_device,				//**ppDevice
		&feature_level_out,			//*pFeatureLevel
		&d3d11_device_context       //**ppImmediateContext
	);
	if (FAILED(hr)) {
		std::cout<<"Create D3D11 device failed. Code:"<<std::hex<<hr<<std::endl;
		return;
	}

	Microsoft::WRL::ComPtr<ID3D11VideoDevice> v_device;
	if (!SUCCEEDED(d3d11_device.As(&v_device))) {
		std::cout << "Create d3d video device failed." << std::endl;
		return;
	}

	// The values here should have _no_ effect on supported profiles, but they
	// are needed anyway for initialization.
	D3D11_VIDEO_PROCESSOR_CONTENT_DESC desc;
	desc.InputFrameFormat = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
	desc.InputFrameRate.Numerator = 60;
	desc.InputFrameRate.Denominator = 1;
	desc.InputWidth = 1920;
	desc.InputHeight = 1080;
	desc.OutputFrameRate.Numerator = 60;
	desc.OutputFrameRate.Denominator = 1;
	desc.OutputWidth = 1920;
	desc.OutputHeight = 1080;
	desc.Usage = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL;

	Microsoft::WRL::ComPtr<ID3D11VideoProcessorEnumerator> enumerator;
	if (!SUCCEEDED(v_device->CreateVideoProcessorEnumerator(&desc, &enumerator))) {
		std::cout << "Create video processor failed." << std::endl;
		return;
	}

	UINT support_flag = 0;
	for (UINT i = 0; i < DXGI_FORMAT_MAX; i++) {
		if (FAILED(enumerator->CheckVideoProcessorFormat(static_cast<DXGI_FORMAT>(i), &support_flag))) {
			std::cout<<"Check video processor format Failed."<<std::endl;
			continue;
		}
		if (support_flag & D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_OUTPUT) 
			std::cout<<"\t"<<DXGI_FORMAT_STRING[i]<<std::endl;
		

	}

}