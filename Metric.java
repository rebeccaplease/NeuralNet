/**
 * For holding metrics in the testing program for each output node
 * and calculating overall accuracy, precision, recall, and f1.
 *
 */
public class Metric {
	double a,b,c,d;
	double overallAccuracy,precision,recall,f1;
	public Metric(){
		a = 0;
		b = 0;
		c = 0;
		d = 0;
	}
	public Metric(double a1, double b1, double c1, int d1){
		a = a1;
		b = b1;
		c = c1;
		d = d1;
	}
	public Metric(double acc, double pre, double rec){
		overallAccuracy = acc;
		precision = pre;
		recall = rec;
	}
	public void calculate(){
		overallAccuracy = (a+d)/(a+b+c+d);
		precision = a/(a+b);
		recall = a/(a+c);
		f1 = (2*precision*recall)/(precision+recall);
	}
	public void calculateMacro(double no){
		overallAccuracy /= no;
		precision /= no;
		recall /= no;
		f1 = (2*precision*recall)/(precision+recall);
	}
	public String abcd(){
		String s = String.format("%.0f %.0f %.0f %.0f ", a,b,c,d);
		return s;
	}
	public String values(){
		String s = String.format("%.3f %.3f %.3f %.3f ", overallAccuracy, precision, recall, f1);
		//System.out.println(s);
		return s;
	}
}
