addpath('/Users/arun/Desktop/Research/Code');
k = 4;
base_dir = '/Users/arun/Desktop/Research/Results/';
runs = {
    'GSE242352', 'SPA1_D'; 'GSE242352', 'SPA2_A'; 'GSE242352', 'SPA5_D';
    'GSE242352', 'SPA7_D'; 'GSE242352', 'SPA8_A';
    'GSE253080', 'GBM1-R1-1'; 'GSE253080', 'GBM1-R1-2'; 'GSE253080', 'GBM1-R1-3'; 'GSE253080', 'GBM1-R1-4';
    'GSE253080', 'GBM1-R2-1'; 'GSE253080', 'GBM1-R2-2'; 'GSE253080', 'GBM1-R2-3'; 'GSE253080', 'GBM1-R2-4';
    'GSE253080', 'GBM2-R1'; 'GSE253080', 'GBM2-R2'; 'GSE253080', 'GBM2-R3'; 'GSE253080', 'GBM2-R4';
    'GSE253080', 'GBM3-R1'; 'GSE253080', 'GBM3-R2'; 'GSE253080', 'GBM3-R3'; 'GSE253080', 'GBM3-R4'
};

for s = 1:size(runs, 1)
    if ~isfile(fullfile(base_dir, runs{s,1}, runs{s,2}, sprintf('%s_norm_counts_tumor.mat', runs{s,2}))), fprintf('skip %s\n', runs{s,2}); continue; end
            
    sec_dir = fullfile(base_dir, runs{s,1}, runs{s,2});
    sample = runs{s,2};
    fprintf('\n>>> Processing %s / %s (k=%d)\n', runs{s,1}, sample, k);

    data = load(fullfile(sec_dir, sprintf('%s_norm_counts_tumor.mat', sample)));
    M = data.M;
    fprintf(' M: %d genes x %d spots\n', size(M,1), size(M,2));

    [C, a, F] = Unmix(M, k);
    fprintf(' a = %.6f | F: %d x %d\n', a, size(F,1), size(F,2));
    fprintf(' F row sums: min=%.4f  max=%.4f\n', min(sum(F,2)), max(sum(F,2)));

    out_path = fullfile(sec_dir, sprintf('%s_tumor_unmix_results.mat', sample));
    save(out_path, 'C', 'a', 'F');
    fprintf(' Saved -> %s\n', out_path);
end

fprintf('\nDone.\n');