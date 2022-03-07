set terminal wxt size 500,500 enhanced font 'Verdana,10' persist

set xlabel 'False Positive Rate (FPR)'
set ylabel 'True Positive Rate (TPR)'
set xrange [0:1]
set yrange [0:1]

f(x) = x
plot 'Debug/roc_data0.dat' title 'ROC1' with lines, \
     'Debug/roc_data1.dat' title 'ROC2' with lines, \
     'Debug/roc_data2.dat' title 'ROC3' with lines, \
     'Debug/roc_data3.dat' title 'ROC4' with lines, \
     'Debug/roc_data4.dat' title 'ROC5' with lines, \
     'Debug/roc_data5.dat' title 'ROC6' with lines, \
     'Debug/roc_data6.dat' title 'ROC7' with lines, \
     'Debug/roc_data7.dat' title 'ROC8' with lines, \
     'Debug/roc_data8.dat' title 'ROC9' with lines, \
     'Debug/roc_data9.dat' title 'ROC10' with lines, \
     'Debug/roc_data10.dat' title 'ROC11' with lines, \
     'Debug/roc_data11.dat' title 'ROC12' with lines, \
     'Debug/roc_data12.dat' title 'ROC13' with lines, \
     'Debug/roc_data13.dat' title 'ROC14' with lines, \
     'Debug/roc_data14.dat' title 'ROC15' with lines, \
     f(x) title 'TPR = FPR' with lines
