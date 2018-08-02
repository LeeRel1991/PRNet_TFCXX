#ifndef SIMPLE_TIMER_H
#define SIMPLE_TIMER_H
#include <string>
#include <sys/time.h>
using std::string;

class SimpleTimer
{
    struct timeval st_tm, end_tm;
    float calcTime(timeval start, timeval end)
    {
        return (1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec) * 1.0 / 1000;
    }

public:
    SimpleTimer(const string &name):
                name_(name)
    {
        gettimeofday(&st_tm, NULL);
    }
    ~SimpleTimer(){
        gettimeofday(&end_tm, NULL);
        printf("\033[36m【%10s】 spent: %.3f ms.\n\033[0m", name_.c_str(), calcTime(st_tm, end_tm));
    }

private:
    string name_;
};



#endif // SIMPLE_TIMER_H
