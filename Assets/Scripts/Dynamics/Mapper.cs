namespace Dynamics
{
    public static class Mapper
    {
        public static IDynamics GetDynamics(DynamicsEnum dynamicsType)
        {
            IDynamics dynamics = dynamicsType switch
            {
                DynamicsEnum.CartesianVelocity => new CartesianVelocity(),
                DynamicsEnum.CartesianAcceleration => new CartesianAcceleration(),
                DynamicsEnum.PolarVelocity => new PolarVelocity(),
                DynamicsEnum.PolarAcceleration => new PolarAcceleration(),
                _ => null
            };

            return dynamics;
        }
    }
}