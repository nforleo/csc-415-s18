#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "tiffio.h"

// saves TIFF file from data in `raster`
void save_tiff(const char *fname, uint32 *raster, uint32 w, uint32 h) {
    TIFF *tif = TIFFOpen(fname, "w");
    if (! raster) {
        throw std::runtime_error("Could not open output file");
    }
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, w);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, h);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 4);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFWriteEncodedStrip(tif, 0, raster, w*h*4);
    TIFFClose(tif);
}

// loads image data from `fname` (allocating dynamic memory)
// *w and *h are updated with the image dimensions
// raster is a matrix flattened into an array using row-major order
// every uint32 in the array is 4 bytes, enconding 8-bit packed ABGR
// A: transparency attribute (can be ignored)
// B: blue pixel
// G: green pixel
// R: red pixel
uint32 *load_tiff(const char *fname, uint32 *w, uint32 *h) {
    TIFF *tif = TIFFOpen(fname, "r");
    if (! tif) {
        throw std::runtime_error("Could not open input file");
    }
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, h);
    uint32 *raster = (uint32 *) _TIFFmalloc(*w * *h * sizeof (uint32));
    if (! raster) {
        TIFFClose(tif);
        throw std::runtime_error("Memory allocation error");
    }
    if (! TIFFReadRGBAImageOriented(tif, *w, *h, raster, ORIENTATION_TOPLEFT, 0)) {
        TIFFClose(tif);
        throw std::runtime_error("Could not read raster from TIFF image");
    }
    TIFFClose(tif);
    return raster;
}

void filter_image_seq(uint32 *raster, uint32 w, uint32 h, const float *filter, int f_len) {
    // to get RGB values from a pixel, you can either use bitwise masks
    // or rely on the following macros:
    // TIFFGetR(raster[i]) red
    // TIFFGetG(raster[i]) green
    // TIFFGetB(raster[i]) blue 
    // TIFFGetA(raster[i]) this value should be ignored
    //
    // to modify RGB values from a pixel, you can use bitwise shifts or masks
    // each pixel stores values in the order ABGR
    // raster[i] = 0xFF0000FF;  // assigns Red to a pixel i
    // raster[i] = 0xFF00FF00;  // assigns Green to a pixel i
    // raster[i] = 0xFFFF0000;  // assigns Blue to a pixel i
    // note that the first byte is always FF (alpha = 255)
    //
    // TODO: here you will filter the image in raster
    //
}

void filter_image_par(uint32 *raster, uint32 w, uint32 h, const float *filter, int f_len, int n_threads) {
    //
    // TODO: here you will filter the image in raster using threads
    //
}

float *load_filter(const char *fname, int *n) {
    std::ifstream myfile(fname);
    if (! myfile) {
        throw std::runtime_error("Could not open filter file");
    }
    myfile >> *n;
    float *filter = new float[*n];
    for (int i = 0 ; i < *n ; i++) myfile >> filter[i];
    myfile.close();
    return filter;
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cout << "Usage:\t./filter <in_fname> <out_fname> <filter_fname> <algo> <n_threads>" << std::endl;
        std::cout << "<in_fname> path to the input image" << std::endl;
        std::cout << "<out_fname> path to the output image" << std::endl;
        std::cout << "<filter_fname> path to the filter file" << std::endl;
        std::cout << "<algo> whether to use the sequential (seq) or parallel algorithm (par)" << std::endl;
        std::cout << "<n_threads> number of threads to use" << std::endl;
        return 0;
    }

    uint32 width, height;
    int n_threads = std::stoi(argv[5]);

    // loads the filter
    int f_len;
    float *filter = load_filter(argv[3], &f_len);
    
    // loads image bytes from file name supplied as a command line argument
    // this function allocates memory dynamically
    uint32 *image = load_tiff(argv[1], &width, &height);

    // measure time of the algorithm
    auto start = std::chrono::high_resolution_clock::now();
    if (! std::strcmp(argv[4], "seq")) {
        // call the sequential implementation
        filter_image_seq(image, width, height, filter, f_len);
    } else if (! std::strcmp(argv[4], "par")) {
        // TODO: call the parallel implementation
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << diff.count() << std::endl;   

    // save new file with filtered image
    save_tiff(argv[2], image, width, height);

    // frees memory allocated by load_filter and load_tiff
    delete [] filter;
    _TIFFfree(image);

    return 0;
}
