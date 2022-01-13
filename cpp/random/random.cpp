/*** 
 * @Author: Lq
 * @Date: 2022-01-04 11:39:35
 * @LastEditTime: 2022-01-04 13:56:51
 * @LastEditors: Lq
 * @Description: 随机数库使用
 */

#include <iostream>
#include <random>

using namespace std;

int main()
{
    random_device rd;

    mt19937 gen(rd());
    std::normal_distribution<float> nd(0.,1.);

    for(int i = 0 ; i < 10 ;i ++) {
        cout << nd(gen) << endl;
    }

    return 0;
}
