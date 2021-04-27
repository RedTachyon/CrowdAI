using System;

namespace Rewards
{
    public static class Mapper
    {
        public static IRewarder GetRewarder(RewardersEnum rewarderType)
        {
            IRewarder rewarder = rewarderType switch
            {
                RewardersEnum.BaseRewarder => new BaseRewarder(),
                _ => null
            };

            return rewarder;
        }
    }
}