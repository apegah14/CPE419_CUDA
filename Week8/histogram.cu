#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <omp.h>

// structure for storing the image data
typedef struct PGMImage {
    char pgmType[3];
    unsigned char** data;
    unsigned int width;
    unsigned int height;
    unsigned int maxValue;
} PGMImage;


// function to ignore any comments in file
void ignoreComments(FILE* fp)
{
    int ch;
    char line[100];
  
    // Ignore any blank lines
    while ((ch = fgetc(fp)) != EOF
           && isspace(ch))
        ;
  
    // Recursively ignore comments
    // in a PGM image commented lines
    // start with a '#'
    if (ch == '#') {
        fgets(line, sizeof(line), fp);
        ignoreComments(fp);
    }
    else
        fseek(fp, -1, SEEK_CUR);
}

// function to read in image
bool openPGM(PGMImage* pgm, const char* filename)
{
    // Open the image file in the
    // 'read binary' mode
    FILE* pgmfile
        = fopen(filename, "rb");
  
    // If file does not exist,
    // then return
    if (pgmfile == NULL) {
        printf("File does not exist\n");
        return false;
    }
  
    ignoreComments(pgmfile);
    fscanf(pgmfile, "%s",
           pgm->pgmType);
  
    // Check for correct PGM Binary
    // file type
    if (strcmp(pgm->pgmType, "P5")) {
        fprintf(stderr,
                "Wrong file type!\n");
        exit(EXIT_FAILURE);
    }
  
    ignoreComments(pgmfile);
  
    // Read the image dimensions
    fscanf(pgmfile, "%d %d",
           &(pgm->width),
           &(pgm->height));
  
    ignoreComments(pgmfile);
  
    // Read maximum gray value
    fscanf(pgmfile, "%d", &(pgm->maxValue));
    ignoreComments(pgmfile);
  
    // Allocating memory to store
    // img info in defined struct
    pgm.data = malloc(pgm->height * sizeof(unsigned char*));
  
    // Storing the pixel info in
    // the struct
    if (pgm->pgmType[1] == '5') {
  
        fgetc(pgmfile);
  
        for (int i = 0; i < pgm->height; i++) {
            pgm->data[i] = malloc(pgm->width * sizeof(unsigned char));
  
            // If memory allocation
            // is failed
            if (pgm->data[i] == NULL) {
                fprintf(stderr,
                        "malloc failed\n");
                exit(1);
            }
  
            // Read the gray values and
            // write on allocated memory
            fread(pgm->data[i],
                  sizeof(unsigned char),
                  pgm->width, pgmfile);
        }
    }
  
    // Close the file
    fclose(pgmfile);
  
    return true;
}

// function to generate histogram on CPU to verify


// function to generate histogram on GPU



int main() {
    PGMImage *pgm = malloc(sizeof(PGMImage));
}