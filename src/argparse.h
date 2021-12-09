#ifndef __ARGPARSE_H__
#define __ARGPARSE_H__

#include <getopt.h>
#include <stdlib.h>
#include <iostream>

struct options_t {
    int num_cluster;
    int dims;
    char *in_file;
    int max_num_iter;
    double threshold;
    bool c;
    int seed;
    bool use_cuda;
};

void get_opts(int argc, char **argv, struct options_t *opts);
#endif
