addpath('/Users/arun/Desktop/Research/Code');
k = 4;
samples = {'G740_ST1', 'G4_ST2'};
base_dir = '/Users/arun/Desktop/Research/Results/';
in_dir  = fullfile(base_dir, 'Unmix_tumor_k4_markers');
out_dir = fullfile(base_dir, 'Unmix_tumor_k4_markers');

for s = 1:length(samples)
    sample = samples{s};
    fprintf('\n>>> Processing %s (k=%d)\n', sample, k);

    in_path = fullfile(in_dir, sprintf('%s_norm_counts_tumor.mat', sample));
    data = load(in_path);
    M = data.M;
    fprintf(' M: %d genes x %d spots\n', size(M,1), size(M,2));

    [C, a, F] = Unmix(M, k);
    fprintf(' a = %.6f | F: %d x %d\n', a, size(F,1), size(F,2));
    fprintf(' F row sums: min=%.4f  max=%.4f\n', min(sum(F,2)), max(sum(F,2)));

    out_path = fullfile(out_dir, sprintf('%s_tumor_unmix_results.mat', sample));
    save(out_path, 'C', 'a', 'F');
    fprintf(' Saved -> %s\n', out_path);
end

fprintf('\nDone.\n');