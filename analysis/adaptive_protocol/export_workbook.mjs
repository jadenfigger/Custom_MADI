// Build the expanded protocol-statistics workbook from the numerical CSV products.
// Runs with Windows Node and @oai/artifact-tool; no scientific arithmetic here.
import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook } from '@oai/artifact-tool';

const output = path.resolve(process.argv[2] ?? 'analysis/adaptive_protocol/outputs/windows_run');
const types = JSON.parse(await fs.readFile(path.join(output, 'table_types.json'), 'utf8'));
const workbook = Workbook.create();
const order = ['protocol_summary', 'top_protocols', 'node_metrics', 'acquisition_columns', 'optimization_history', 'protocol_sweeps', 'definitions_config'];
const leads = {
  protocol_summary: ['protocol_name','total_measurements','amplitude_model','robust_score','identifiable_fraction','TE_A_s','sigma_single','protocol_id','objective','objective_value'],
  top_protocols: ['budget','rank','objective','coverage','TE_A_s','sigma_single','protocol_id'],
  node_metrics: ['protocol_id','amplitude_model','node_index','actual_rho','actual_V','actual_k_io','identifiable','invalidity_reason','relative_crlb_sd_log_rho','relative_crlb_sd_log_V','relative_crlb_sd_k_io'],
  acquisition_columns: ['protocol_name','delta_ms','Delta_ms','b_s_mm2','averages','gradient_T_m','TE_A_s','sigma_single','full_column','protocol_id'],
  optimization_history: ['budget','evaluation_id','stage','score','feasible','coverage','best_so_far','cache_hit','TE_A_s','failure'],
  protocol_sweeps: ['budget','sweep_kind','score','coverage','TE_A_s','delta_ms','Delta_ms','b_s_mm2','allocation'],
  definitions_config: ['field','definition','run_id'],
};

// RFC-4180 parser retains literal IDs and embedded JSON. Types come from the
// Python export's schema, so +inf remains explicit text and numeric cells typed.
function parseCSV(text) {
  text = text.replace(/^\uFEFF/,'');
  const rows=[]; let row=[], value='', quoted=false;
  for (let i=0; i<text.length; i++) {
    const c=text[i];
    if (c==='"') {
      if (quoted && text[i+1]==='"') { value+='"'; i++; }
      else quoted=!quoted;
    } else if (c===',' && !quoted) { row.push(value); value=''; }
    else if (c==='\n' && !quoted) { row.push(value.replace(/\r$/,'')); rows.push(row); row=[]; value=''; }
    else value+=c;
  }
  if (value || row.length) { row.push(value.replace(/\r$/,'')); rows.push(row); }
  if (quoted) throw new Error('Unterminated CSV quoted field');
  return rows;
}
const audit=[];
for (const name of order) {
  const matrix=parseCSV(await fs.readFile(path.join(output, `${name}.csv`),'utf8'));
  const inputHeader=matrix.shift();
  const header=[...leads[name].filter(x=>inputHeader.includes(x)), ...inputHeader.filter(x=>!leads[name].includes(x))];
  const positions=header.map(x=>inputHeader.indexOf(x));
  const rows=matrix.map(row=>positions.map((p,j)=>{
    const v=row[p] ?? '';
    if (v==='') return null;
    const type=types[name][header[j]];
    if (type==='number' && Number.isFinite(Number(v))) return Number(v);
    if (type==='boolean') return v.toLowerCase()==='true';
    if (v.length>32767) throw new Error(`Excel cell limit exceeded: ${name}/${header[j]}`);
    return v.startsWith('=') ? `'${v}` : v;
  }));
  const sheet=workbook.worksheets.add(name);
  sheet.showGridLines=false;
  sheet.getRangeByIndexes(0,0,1,header.length).values=[header];
  for(let i=0; i<rows.length; i+=250) {
    const block=rows.slice(i,i+250);
    sheet.getRangeByIndexes(i+1,0,block.length,header.length).values=block;
  }
  const all=sheet.getRangeByIndexes(0,0,rows.length+1,header.length);
  all.format.font={name:'Arial',size:10};
  all.format.columnWidth=24;
  const head=sheet.getRangeByIndexes(0,0,1,header.length);
  head.format={fill:'#244466',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:64,verticalAlignment:'center'};
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(name==='definitions_config'?1:2);
  const table=sheet.tables.add(all,true,`Adaptive_${name}`);
  table.showFilterButton=true;
  for(let j=0;j<header.length;j++) {
    const key=header[j];
    const col=sheet.getRangeByIndexes(1,j,Math.max(rows.length,1),1);
    if(types[name][key]==='number') {
      col.setNumberFormat(/fraction|coverage/.test(key)?'0.0%':/count|index|measurements|^averages$|^rank$|^budget$|^evaluation_id$|^n_|^full_column$/.test(key)?'#,##0':'0.0000E+00');
    }
    if(/protocol_name|amplitude_model|sweep_kind|stage|invalidity_reason|failure/.test(key)) col.format.columnWidth=31;
    if(/json|formula/.test(key)) col.format.columnWidth=45;
  }
  if(name==='definitions_config') {
    sheet.getRangeByIndexes(0,0,rows.length+1,1).format.columnWidth=34;
    sheet.getRangeByIndexes(0,1,rows.length+1,1).format.columnWidth=120;
    sheet.getRangeByIndexes(1,1,rows.length,1).format.wrapText=true;
    for(let i=0;i<rows.length;i++) {
      const isJSON=String(rows[i][0]).includes('.json.');
      sheet.getRangeByIndexes(i+1,0,1,header.length).format.rowHeight=isJSON?45:Math.max(32,Math.ceil(String(rows[i][1]).length/110)*16);
    }
  }
  audit.push({sheet:name,rows:rows.length,columns:header.length});
  console.log(`Loaded ${name}: ${rows.length} x ${header.length}`);
}
workbook.recalculate();
await fs.mkdir(path.join(output,'workbook_previews'),{recursive:true});
for(const name of order) {
  const preview=await workbook.render({sheetName:name,range:name==='definitions_config'?'A1:B9':'A1:F9',scale:1.3,format:'png'});
  await fs.writeFile(path.join(output,'workbook_previews',`${name}.png`),new Uint8Array(await preview.arrayBuffer()));
}
console.log((await workbook.inspect({kind:'table',range:'top_protocols!A1:F7',include:'values,formulas',tableMaxRows:7,tableMaxCols:6,maxChars:3000})).ndjson);
console.log((await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!',options:{useRegex:true,maxResults:10},maxChars:1500})).ndjson);
const exported=await workbook.export({format:'xlsx'});
const temporary=path.join(output,'protocol_statistics.tmp.xlsx');
await fs.writeFile(temporary, new Uint8Array(await exported.arrayBuffer()));
await fs.rename(temporary,path.join(output,'protocol_statistics.xlsx'));
await fs.writeFile(path.join(output,'workbook_audit.json'),JSON.stringify({sheets:audit,checks:'Typed numerical values, bounded previews of every sheet, formula error scan; static scientific data snapshot.'},null,2));
console.log(`Saved ${path.join(output,'protocol_statistics.xlsx')}`);
