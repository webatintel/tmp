#include "DxVideoInfoChecker.h"

int main() {
	DxVideoInfoChecker video_info_checker;
	video_info_checker.CheckOverlayCapability();
	video_info_checker.CheckVideoProcessorCapability();

	system("pause");
	return 0;
}
