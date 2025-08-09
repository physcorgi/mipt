# report.py
from fpdf import FPDF
import tempfile, os, json

class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, 'Terrain minima clustering report', ln=1, align='C')
        self.ln(2)

def build_pdf_report(title, parameters, summary_tables, image_files_bytes):
    """
    parameters: dict
    summary_tables: dict of pandas.DataFrame
    image_files_bytes: list of tuples (filename, bytes)
    returns pdf bytes
    """
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, f"Title: {title}", ln=1)
    pdf.ln(2)
    pdf.cell(0, 6, "Parameters:", ln=1)
    for k,v in parameters.items():
        pdf.cell(0, 6, f"- {k}: {v}", ln=1)
    pdf.ln(4)

    for tname, df_table in summary_tables.items():
        pdf.set_font("Helvetica", 'B', 11)
        pdf.cell(0, 6, tname, ln=1)
        pdf.set_font("Helvetica", size=9)
        pdf.ln(1)
        cols = list(df_table.columns)
        # header
        for c in cols:
            text = str(c)[:15]
            pdf.cell(40, 6, text, border=1)
        pdf.ln()
        for i, row in df_table.head(10).iterrows():
            for c in cols:
                val = row[c]
                txt = (f"{val:.3f}" if isinstance(val, float) else str(val))[:15]
                pdf.cell(40, 6, txt, border=1)
            pdf.ln()
        pdf.ln(4)

    for fname, bts in image_files_bytes:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(fname)[1])
        tmp.write(bts); tmp.flush(); tmp.close()
        pdf.add_page()
        pdf.set_font("Helvetica", 'B', 11)
        pdf.cell(0,6, fname, ln=1)
        pdf.image(tmp.name, x=15, w=180)
        os.unlink(tmp.name)
    return pdf.output(dest='S').encode('latin-1')