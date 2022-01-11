
#include <vector>
#include <climits>

using namespace std;

int maxSubArray(vector<int> &nums)
{
    int max = INT_MIN;
    for(auto it  = nums.begin();it < nums.end(); it ++){
        int sum = 0;
        for(auto it2 = it; it2 < nums.end(); it2 ++){
            sum += *it2;
            if(sum > max){
                max = sum;
            }
        }
    }

    return max;
}

int maxSubArray2(vector<int> &nums){

    int max = nums[0];

    // 以index=i为末尾的连续子数组的最大和
    vector<int> dp(nums.size());
    for(size_t i = 1; i < nums.size(); i ++){

        // 计算以i为结尾的子数组的最大和
        // 以i为结尾的子数组最大和有两种情况，
        // 1. 将nums[i]累加到 dp[i-1]上
        // 2. nums[i]的值比 dp[i - 1] + nums[i]大，则dp[i] = nums[i]
        dp[i] = dp[i - 1] + nums[i];
        dp[i] = std::max(dp[i],nums[i]);

        // 更新子数组的最大值
        max = std::max(max,dp[i]);
    }

    return max;
}

int maxSubArray3(vector<int> &nums)
{
    int max = nums[0];
    int sum = 0;

    /*
        以当前的i为主体，如果之前的sum对当前i成为最大和有增益就加上
        如果，之前的sum对当前的i成为最大和没有增益则舍弃，并且重新计算
    */
    for(size_t i = 0; i < nums.size(); i ++){
        if(sum > 0){
            sum += nums[i];
        }else {
            sum = nums[i];
        }

        max = std::max(sum,max);
    }

    return max;
}

int maxSubArray4(vector<int> &nums)
{
    int max = nums[0];

    int sum = 0; // 当前子数组的和
    
    for(size_t i = 0; i < nums.size(); i ++){
        
        // 之前的和为sum，加上当前的nums[i]后有两种情况
        if(nums[i] > sum + nums[i]){
            sum = nums[i]; // 则当前最大的和nums[i]
        } else {
            sum = sum + nums[i];
        }

        max = std::max(max,sum);
    }

    return max;
}

int main()
{
    return 0;
}