set term postscript eps enhanced color
set output "10_ri.eps"
set size ratio 1
set yrange [0:*]
set xlabel "Index"
set ylabel "Rotation Error [deg/m]"
plot "10_ri.txt" using 1:($2*57.3) title 'Rotation Error' lc rgb "#0000FF" pt 2 w lines
