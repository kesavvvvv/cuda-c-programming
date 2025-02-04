__global__
void imageBlurKernel(uchar4* input, uchar4* output, int width, int height) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < width && row < height) {
        
        int curPixel = row * width + col;
        
        int rSum = 0, gSum = 0, bSum = 0, numPixels = 0;

        for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                int i = curRow * width + curCol;

                if(curRow >= 0 && curRow < height && curCol >=0 && curCol < width) {
                    rSum += input[i].x;
                    gSum += input[i].y;
                    bSum += input[i].z;
                    ++numPixels;
                }
            }
        }

        output[curPixel].x = rSum / pixels;
        output[curPixel].y = gSum / pixels;
        output[curPixel].z = bSum / pixels;
    }
}

void initializeImage(const string& imagePath, uchar4 * input, int *width, int *height) {
    // Load the image using OpenCV
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return nullptr;
    }

    // Get image dimensions
    *width = image.cols;
    *height = image.rows;

    // Allocate memory for uchar4 array
    input = (uchar4*)malloc(width * height * sizeof(uchar4));


    // Copy RGB values into uchar4 array
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
            int idx = i * width + j;
            input[idx].x = pixel[2];  // R
            input[idx].y = pixel[1];  // G
            input[idx].z = pixel[0];  // B
            input[idx].w = 255;       // Alpha (Opaque)
        }
    }

}

int main() {
    int width, height;
    uchar4 * input;

    initializeImage("image.jpg", input, &width, &height);

    uchar4 * output_d;
    uchar4 * output;

    uchar4 * input_d;

    cudaMalloc(&input_d, sizeof(uchar4)*width*height);

    cudaMalloc(&output_d, sizeof(uchar4)*width*height);

    cudaMemcpy(input_d, input, sizeof(uchar4)*width*height, cudaMemcpyHostToDevice);

    dim3 dimGrid = {ceil(width/16.0), ceil(height/16.0)};
    dim3 dimBlock = {16, 16};

    imageBlurKernel<<<dimGrid, dimBlock>>>(input_d, output_d, width, height);

    cudaMemcpy(output, output_d, sizeof(uchar4)*width*height, cudaMemcpyDeviceToHost);

    cudaFree(output_d);
    cudaFree(input_d);
    
}

