#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include <stack>
#include <fstream>
#include <map>
#include <math.h>
#include <net.h>
#include "prompt_slover.h"
#include "decoder_slover.h"
#include "diffusion_slover.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <time.h>
#include "getmem.h"
using namespace std;

int main()
{
	//ncnn::Mat input(512, 512, 128), out, pout;
	//input.fill(1.2f);

	//{
	//	ncnn::Net conv;
	//	conv.opt.use_vulkan_compute = false;
	//	conv.opt.lightmode = true;
	//	conv.opt.use_winograd_convolution = false;
	//	conv.opt.use_sgemm_convolution = false;
	//	conv.opt.use_fp16_packed = true;
	//	conv.opt.use_fp16_storage = true;
	//	conv.opt.use_fp16_arithmetic = true;
	//	conv.opt.use_packing_layout = true;
	//	conv.load_param("assets/pConv/conv.param");
	//	conv.load_model("assets/pConv/conv.bin");
	//	printf(" %.2lfM\n", getCurrentRSS() / 1024.0 / 1024.0);
	//	{
	//		ncnn::Extractor ex = conv.create_extractor();
	//		ex.set_light_mode(true);
	//		ex.input("in0", input);
	//		ex.extract("out0", out);
	//	}
	//	printf(" %.2lfM\n", getCurrentRSS() / 1024.0 / 1024.0);
	//}

	//{
	//	ncnn::Net pconv;
	//	pconv.opt.use_vulkan_compute = false;
	//	pconv.opt.lightmode = true;
	//	pconv.opt.use_winograd_convolution = false;
	//	pconv.opt.use_sgemm_convolution = false;
	//	pconv.opt.use_fp16_packed = true;
	//	pconv.opt.use_fp16_storage = true;
	//	pconv.opt.use_fp16_arithmetic = true;
	//	pconv.opt.use_packing_layout = true;
	//	pconv.register_custom_layer("PConvolution", PConvolution_layer_creator);
	//	pconv.load_param("assets/pConv/pconv.param");
	//	pconv.load_model("assets/pConv/pconv.bin");
	//	printf(" %.2lfM\n", getCurrentRSS() / 1024.0 / 1024.0);
	//	{
	//		ncnn::Extractor ex = pconv.create_extractor();
	//		ex.set_light_mode(true);
	//		ex.input("in0", input);
	//		ex.extract("out0", pout);
	//	}
	//	printf(" %.2lfM\n", getCurrentRSS() / 1024.0 / 1024.0);
	//}

	//{
	//	if (out.c == pout.c && out.h == pout.h && out.w == pout.w) {
	//		float diff = 0.f;
	//		float sum = 0.f;
	//		for (int c = 0; c < out.c; c++) {
	//			for (int hw = 0; hw < out.h * out.w; hw++) {
	//				sum += std::abs(out.channel(c)[hw]);
	//				diff += std::abs(out.channel(c)[hw] - pout.channel(c)[hw]);
	//			}
	//		}
	//		printf("sum:%f, diff:%f\n", sum, diff);
	//	}
	//	else {
	//		std::cout << "dimension error!" << std::endl;
	//	}
	//}


	
	int height, width, mode, step, seed;
	string positive_prompt, negative_prompt;

	// default setting
	height = 256;
	width = 256;
	mode = 0;
	step = 15;
	seed = 42;
	positive_prompt = "floating hair, portrait, ((loli)), ((one girl)), cute face, hidden hands, asymmetrical bangs, beautiful detailed eyes, eye shadow, hair ornament, ribbons, bowties, buttons, pleated skirt, (((masterpiece))), ((best quality)), colorful";
	negative_prompt = "((part of the head)), ((((mutated hands and fingers)))), deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, Octane renderer, lowres, bad anatomy, bad hands, text";

	// parse the magic.txt
	ifstream magic;
	magic.open("magic.txt");
	if (!magic) {
		cout << "can not find magic.txt, using the default setting" << endl;
	}
	else {
		string content = "";
		int i = 0;
		for (i = 0; i < 7; i++) {
			if (getline(magic, content)) {
				switch (i)
				{
				case 0:height = stoi(content);
				case 1:width = stoi(content);
				case 2:mode = stoi(content);
				case 3:step = stoi(content);
				case 4:seed = stoi(content);
				case 5:positive_prompt = content;
				case 6:negative_prompt = content;
				default:break;
				}
			}
			else {
				break;
			}
		}
		if (i != 7) {
			cout << "magic.txt has wrong format, please fix it" << endl;
			return 0;
		}

	}
	if (seed == 0) {
		seed = (unsigned)time(NULL);
	}
	magic.close();

	// stable diffusion
	cout << "----------------[init]--------------------";
	PromptSlover prompt_slover;
	DiffusionSlover diffusion_slover(height, width, mode);
	DecodeSlover decode_slover(height, width, diffusion_slover._PConvolution_layer_creator);
	printf(" %.2lfG / %.2lfG\n", getCurrentRSS() / 1024.0 / 1024.0 / 1024.0, getPeakRSS() / 1024.0 / 1024.0 / 1024.0);

	cout << "----------------[prompt]------------------";
	ncnn::Mat cond = prompt_slover.get_conditioning(positive_prompt);
	ncnn::Mat uncond = prompt_slover.get_conditioning(negative_prompt);
	printf(" %.2lfG / %.2lfG\n", getCurrentRSS() / 1024.0 / 1024.0 / 1024.0, getPeakRSS() / 1024.0 / 1024.0 / 1024.0);

	cout << "----------------[diffusion]---------------" << endl;
	ncnn::Mat sample = diffusion_slover.sampler(seed, step, cond, uncond);
	cout << "----------------[diffusion]---------------";
	printf(" %.2lfG / %.2lfG\n", getCurrentRSS() / 1024.0 / 1024.0 / 1024.0, getPeakRSS() / 1024.0 / 1024.0 / 1024.0);

	cout << "----------------[decode]------------------";
	ncnn::Mat x_samples_ddim = decode_slover.decode(sample);
	printf(" %.2lfG / %.2lfG\n", getCurrentRSS() / 1024.0 / 1024.0 / 1024.0, getPeakRSS() / 1024.0 / 1024.0 / 1024.0);

	cout << "----------------[save]--------------------" << endl;
	cv::Mat image(height, width, CV_8UC3);
	x_samples_ddim.to_pixels(image.data, ncnn::Mat::PIXEL_RGB2BGR);
	cv::imwrite("result_" + to_string(step) + "_" + to_string(seed) + "_" + to_string(height) + "x" + to_string(width) + ".png", image);

	cout << "----------------[close]-------------------" << endl;
	
	


	return 0;
}
