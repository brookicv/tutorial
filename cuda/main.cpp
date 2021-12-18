/*** 
 * @Author: Lq
 * @Date: 2021-12-17 18:39:03
 * @LastEditTime: 2021-12-17 18:39:03
 * @LastEditors: Lq
 * @Description: 
 */

#include "fmt/format.h"
#include <iostream>

using namespace std;

int main()
{
    fmt::print("hello world\n");
    auto a = fmt::format("{},{}",10,"Shiledon");
    cout << a << endl;
    return 0;
}
