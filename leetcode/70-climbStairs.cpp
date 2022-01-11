/*** 
 * @Author: Lq
 * @Date: 2022-01-10 15:44:44
 * @LastEditTime: 2022-01-10 16:43:49
 * @LastEditors: Lq
 * @Description: 
 */

#include <vector>
#include <iostream>
using namespace std;

int climbStaris(int n)
{
    if(n == 0) return 0;
    if(n == 1) return 1;
    if(n == 2) return 2;

    vector<int> dp(n+1);
    dp = {0};

    dp[1] = 1;
    dp[2] = 2;
    int index = 3;
    while(index <= n){
        dp[index] = dp[index -1] +dp[index - 2];
        index ++;
    }

    return dp[n];
}


int main()
{
    int n;
    cin >> n;
    cout << climbStaris(n) << endl;

    return 0;
}