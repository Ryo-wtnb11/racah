% Export QSpace generator matrices (weights Z, ladder Sp_k, Cartan Sz_k)
% at double precision via the patched getRC MEX, in QSpace's own irrep basis.
function gen_qspace_generators(outfile)
  cases = { {'SO5','[1 0]'}, {'SO5','[0 2]'}, {'Sp4','[1 0]'}, {'Sp4','[0 1]'}, {'Sp4','[2 0]'}, ...
            {'SO6','[1 0 0]'}, {'SO6','[0 1 1]'} };
  fid=fopen(outfile,'w');
  fprintf(fid,'--- QSpace generator export (basis = QSpace RSet, via patched getRC, plain double)\n');
  fprintf(fid,'--- source: QSpace v4-pub @ dd2cc7e; convention: Sz{k} diagonal Cartan (weight coords), Sp{k} ladder ops; [Sp_k,Sp_k^T] closes on integer combos of Sz\n');
  fprintf(fid,'--- format: IRREP sym | J | dim | nops ; Z rows ; OP SP|SZ k: "row col value" 0-based dense-nonzeros\n');
  for c=1:numel(cases)
    R=getRC(cases{c}{1}, cases{c}{2});
    d=size(R.Z,1); n=numel(R.Sp);
    fprintf(fid,'IRREP %s | %s | dim=%d | nops=%d\n', cases{c}{1}, num2str(R.J), d, n);
    for r=1:d, fprintf(fid,'Z %s\n', num2str(R.Z(r,:),'%g ')); end
    for k=1:n
      for OP={'SP','SZ'}
        if strcmp(OP{1},'SP'), M=R.Sp{k}; else, M=R.Sz{k}; end
        [ri,ci,v]=find(M);
        fprintf(fid,'OP %s %d nnz=%d\n', OP{1}, k, numel(v));
        for j=1:numel(v), fprintf(fid,'%d %d %.17g\n', ri(j)-1, ci(j)-1, v(j)); end
      end
    end
    fprintf('done %s %s dim=%d\n', cases{c}{1}, cases{c}{2}, d);
  end
  fclose(fid);
end
