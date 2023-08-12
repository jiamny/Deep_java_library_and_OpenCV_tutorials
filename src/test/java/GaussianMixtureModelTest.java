import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.jiamny.Utils.DeterminantCalc;
import org.testng.annotations.Test;

import java.util.Arrays;

import static com.jiamny.Utils.UtilFunctions.ShuffleArray;

public class GaussianMixtureModelTest {

    @Test
    public void testGMM() {

        int [] elements = new int[5];
        for(int i = 0; i < 5; i++)
            elements[i] = i;
        System.out.println(Arrays.toString(elements));

        for(int i = 0; i < 5; i++) {
            elements = ShuffleArray(elements);
            System.out.println(Arrays.toString(elements));
        }

        NDManager manager = NDManager.newBaseManager();
        int n_samples = 10, k = 5;
        NDArray likelihoods_cls = manager.zeros(new Shape(n_samples, k));
        NDArray nd = manager.arange(10);
        likelihoods_cls.set(new NDIndex(":,0"), nd);
        System.out.println(likelihoods_cls);


        double [][] tt = new double[][]{
                {0.7037, -0.0811,  1.2845,  0.5050},
                {-0.0811,  0.2312, -0.4393, -0.1712},
                {1.2845, -0.4393,  2.9398,  1.1616},
                {0.5050, -0.1712,  1.1616,  0.4793}
        };

        // determinant  tensor(0.0007)

        NDArray cov = manager.create(tt).toType(DataType.FLOAT64, false);
        int rows = 4, cols = 4;
        double [][] t = new double[rows][cols];
        for( int r = 0; r < rows; r++)
            t[r] = (double[])Arrays.copyOf(cov.get(r).toDoubleArray(), cols);

        System.out.println(DeterminantCalc.Determinant(t));
        NDArray determinant = manager.create(DeterminantCalc.Determinant(t));
        NDArray dim = manager.create(4);
        double p = Math.pow(2.0 * Math.PI, 4);
        NDArray a =  manager.create(p).mul(determinant).sqrt();
        System.out.println(a);
        System.out.println(NDArrays.div(1.0, a));
    }
}
