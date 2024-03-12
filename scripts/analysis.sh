METHOD=$1
for thresh in 0.0 0.1 0.3 0.5 0.9 1.0;
    do
    # nohup bash scripts/template.sh ${METHOD} txt ${thresh} 1 >logs/out_files/clip/${METHOD}_thresh${thresh}_analysis.out
    nohup bash scripts/template_maple.sh ${METHOD} txt ${thresh} 2 >logs/out_files/maple/analysis/${METHOD}_thresh${thresh}_analysis.out
    done