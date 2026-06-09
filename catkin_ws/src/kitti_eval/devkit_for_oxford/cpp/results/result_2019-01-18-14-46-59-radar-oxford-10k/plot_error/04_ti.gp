set term postscript eps enhanced color
set output "04_ti.eps"
set size ratio 1
set yrange [0:*]
set xlabel "Index"
set ylabel "Translation Error [%]"
plot "04_ti.txt" using 1:($2*100) title 'Translation Error' lc rgb "#0000FF" pt 2 w lines
