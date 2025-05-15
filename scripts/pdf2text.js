const fs = require('fs');
const path = require('path');
const pdf = require('pdf-parse');

async function convertPdfToText(pdfPath, outputDir) {
    try {
        // Read the PDF file
        const dataBuffer = fs.readFileSync(pdfPath);
        
        // Parse the PDF
        const data = await pdf(dataBuffer);
        
        // Create output filename
        const filename = path.basename(pdfPath, '.pdf') + '.txt';
        const outputPath = path.join(outputDir, filename);
        
        // Write the text content to a file
        fs.writeFileSync(outputPath, data.text);
        
        console.log(`Successfully converted ${pdfPath} to ${outputPath}`);
    } catch (error) {
        console.error(`Error converting ${pdfPath}:`, error);
    }
}

async function main() {
    const knowledgeDir = path.join(process.cwd(), 'characters', 'knowledge');
    const outputDir = path.join(process.cwd(), 'agent', 'Knowledge');
    
    // Create output directory if it doesn't exist
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }
    
    // Get all PDF files in the knowledge directory
    const files = fs.readdirSync(knowledgeDir);
    const pdfFiles = files.filter(file => file.endsWith('.pdf'));
    
    // Convert each PDF file
    for (const pdfFile of pdfFiles) {
        const pdfPath = path.join(knowledgeDir, pdfFile);
        await convertPdfToText(pdfPath, outputDir);
    }
}

main().catch(console.error); 