% Generate QSpace CGC oracle fixtures for racah's B/C/D (and SU2) channels.
%
% Requires: getCG.mexmaca64 built from QSpace v4-pub @ dd2cc7e with the
% maca64 API-sync patch (tools/qspace_getCG_maca64.patch); RC_STORE env set.
% Output format per record:
%   CH <sym> | <J1> | <J2> | <J> | <size...>
%   <idx row (0-based)> <value>   (one line per nonzero)
% Normalization: QSpace stores CGC at unit Frobenius norm per (channel, OM).
function gen_qspace_cgc(outfile)
  cases = { ...
    {'SU2',[1],[1]}, {'SU2',[2],[2]}, {'SU2',[3],[2]}, ...
    {'SO5',[1 0],[1 0]}, {'SO5',[0 2],[0 2]}, {'SO5',[1 0],[0 2]}, ...
    {'Sp4',[1 0],[1 0]}, {'Sp4',[0 1],[0 1]}, {'Sp4',[1 0],[0 1]}, ...
    {'SO6',[1 0 0],[1 0 0]}, {'SO6',[0 1 1],[0 1 1]}, {'SO6',[1 0 0],[0 1 1]}, ...
    {'SO7',[1 0 0],[1 0 0]}, {'Sp6',[1 0 0],[1 0 0]}, {'SO8',[1 0 0 0],[1 0 0 0]} };
  fid = fopen(outfile,'w');
  fprintf(fid,'--- racah QSpace CGC oracle fixtures\n');
  fprintf(fid,'--- tool: QSpace v4-pub @ dd2cc7e + maca64 sync patch; MATLAB %s\n', version);
  fprintf(fid,'--- normalization: unit Frobenius per (channel, OM slice)\n');
  fprintf(fid,'--- format: CH sym | J1 | J2 | J | size ; then "i j k [m] value" 0-based\n');
  for c = 1:numel(cases)
    sym=cases{c}{1}; J1=cases{c}{2}; J2=cases{c}{3};
    T=getCG(sym,J1,J2); r=numel(J1);
    for i=1:numel(T)
      q=T(i).qset; J=q(2*r+1:end); sz=T(i).size;
      fprintf(fid,'CH %s | %s | %s | %s | %s\n', sym, num2str(J1), num2str(J2), num2str(J), num2str(sz));
      idx=T(i).idx; dat=T(i).data;
      for k=1:numel(dat)
        fprintf(fid,'%s %.17g\n', num2str(idx(k,:)), dat(k));
      end
    end
    fprintf('done %s [%s]x[%s]: %d channels\n', sym, num2str(J1), num2str(J2), numel(T));
  end
  fclose(fid);
end
