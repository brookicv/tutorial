/*** 
 * @Author: Lq
 * @Date: 2022-01-07 11:26:11
 * @LastEditTime: 2022-01-07 14:37:19
 * @LastEditors: Lq
 * @Description: 简单的计时器
 */

#ifndef __LQ_TIMER_INCLUDE_
#define __LQ_TIMER_INCLUDE_


#include <chrono>
#include <string>

class Timer {
public:
    Timer();
    explicit Timer(const std::string &name);

    void start();
    float elapsed();

private:
    std::string m_name;
    std::chrono::steady_clock::time_point m_start_time;
    std::chrono::steady_clock::time_point m_end_time;
};

Timer::Timer():m_name("Time Elapsed:")
{
    start();
}

Timer::Timer(const std::string &name):m_name(name)
{
    start();
}


void Timer::start()
{
    m_start_time = std::chrono::steady_clock::now();
}

float Timer::elapsed()
{
    m_end_time = std::chrono::steady_clock::now();
    std::chrono::duration<float,std::milli> duration = m_end_time - m_start_time;

    return duration.count();
}


#endif
