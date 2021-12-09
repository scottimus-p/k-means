#include <argparse.h>

#include <cstring>

using std::strcmp;

void get_opts(int argc,
              char **argv,
              struct options_t *opts)
{
    if (argc == 1)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t-k <number of clusters>" << std::endl;
        std::cout << "\t-d <dimension of the points>" << std::endl;
        std::cout << "\t-i <file_path>" << std::endl;
        std::cout << "\t-m <maximum number of iterations>" << std::endl;
        std::cout << "\t-t <threshold for convergence test>" << std::endl;
        std::cout << "\t[Optional] -c <flag to output centroids of all clusters>" << std::endl;
        std::cout << "\t[Optional] --cuda <flag to use CUDA>" << std::endl;
        std::cout << "\t-s <seed for rand()>" << std::endl;

        exit(0);
    }

    opts->c = false;
    opts->use_cuda = false;

    struct option l_opts[] = {
        {"num-clusters", required_argument, NULL, 'k'},
        {"data-dimension", required_argument, NULL, 'd'},
        {"input-file", required_argument, NULL, 'i'},
        {"max-iter", required_argument, NULL, 'm'},
        {"threshold", required_argument, NULL, 't'},
        {"cuda", no_argument, NULL, '\0'},
        {"seed", required_argument, NULL, 's'}
    };

    int ind, c;
    while ((c = getopt_long(argc, argv, "k:d:i:m:t:cs:", l_opts, &ind)) != -1)
    {
        switch (c)
        {
        case 0:
            if (l_opts[ind].flag != 0)
                break;
            if (strcmp(l_opts[ind].name,"cuda") == 0)
                opts->use_cuda = true;
            break;
        case 'k':
            opts->num_cluster = atoi((char *) optarg);
            break;
        case 'd':
            opts->dims = atoi((char *) optarg);
            break;
        case 'i':
            opts->in_file = (char *)optarg;
            break;
        case 'm':
            opts->max_num_iter = atoi((char *) optarg);
            break;
        case 't':
            opts->threshold = atof((char *) optarg);
            break;
        case 'c':
            opts->c = true;
            break;
        case 's':
            opts->seed = atoi((char *) optarg);
            break;
        case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(1);
        }
    }
}