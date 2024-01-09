#include "utils/io.h"
#include "utils/points.h"

#include "ceres/ceres.h"
#include <math.h>

struct RegistrationCostFunction
{
	RegistrationCostFunction(const Point2D& point_1, const Point2D& point_2)
			: point_1(point_1), point_2(point_2)
		{
		}

		template<typename T>
		bool operator()(const T* const angle, const T* const tx, const T* const ty, T* residual) const
		{
			// TODO: Implement the cost function
			residual[0] = point_2.x - (cos(angle[0]) * point_1.x - sin(angle[0]) * point_1.y + tx[0]);
			residual[1] = point_2.y - (sin(angle[0]) * point_1.x + cos(angle[0]) * point_1.y + ty[0]);
			return true;
		}

	private:
		const Point2D point_1, point_2;
};

