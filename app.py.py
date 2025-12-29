from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile

app = Flask(__name__)
CORS(app)  # للسماح بالاتصال من الإضافة

# إعدادات OpenAI
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# مجلد قاعدة البيانات
CHROMA_DIR = "./chroma_db"

# متغير عام لحفظ النظام
qa_system = None

# رسالة النظام بالعربية
SYSTEM_PROMPT = """أنت مساعد ذكي ومفيد. مهمتك الإجابة على الأسئلة بناءً على المعلومات المتوفرة في المستندات فقط.

القواعد المهمة:
- أجب باللغة العربية دائماً
- استخدم فقط المعلومات الموجودة في السياق
- إذا لم تجد الإجابة في المستندات، قل "عذراً، لا أجد هذه المعلومة في المستندات المتاحة"
- كن واضحاً ومختصراً
- إذا كانت المعلومات ناقصة، اذكر ذلك

السياق المتاح:
{context}

السؤال: {question}

الإجابة:"""


def initialize_db():
    """تحميل قاعدة البيانات إذا كانت موجودة"""
    global qa_system
    
    if os.path.exists(CHROMA_DIR):
        print("📂 تحميل قاعدة البيانات الموجودة...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        
        # إنشاء نظام الأسئلة والأجوبة
        llm = ChatOpenAI(model="gpt-4-mini", temperature=0)
        
        prompt = PromptTemplate(
            template=SYSTEM_PROMPT,
            input_variables=["context", "question"]
        )
        
        qa_system = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        print("✅ النظام جاهز!")
        return True
    
    print("⚠️ قاعدة البيانات غير موجودة")
    return False


@app.route('/')
def home():
    """الصفحة الرئيسية"""
    return jsonify({
        "status": "online",
        "message": "🤖 RAG Chatbot API يعمل بنجاح!",
        "endpoints": {
            "/chat": "POST - للمحادثة",
            "/upload": "POST - لرفع ملف PDF جديد",
            "/status": "GET - للتحقق من حالة النظام"
        }
    })


@app.route('/status')
def status():
    """التحقق من حالة النظام"""
    db_exists = os.path.exists(CHROMA_DIR)
    system_ready = qa_system is not None
    
    return jsonify({
        "database_exists": db_exists,
        "system_ready": system_ready,
        "message": "النظام جاهز ✅" if system_ready else "يرجى رفع ملف PDF أولاً ⚠️"
    })


@app.route('/chat', methods=['POST'])
def chat():
    """نقطة نهاية المحادثة"""
    global qa_system
    
    if qa_system is None:
        return jsonify({
            "error": "النظام غير جاهز. يرجى رفع ملف PDF أولاً",
            "ready": False
        }), 400
    
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({"error": "السؤال مطلوب"}), 400
    
    try:
        # الحصول على الإجابة
        result = qa_system({"query": question})
        
        return jsonify({
            "answer": result['result'],
            "sources": len(result.get('source_documents', [])),
            "success": True
        })
    
    except Exception as e:
        return jsonify({
            "error": f"حدث خطأ: {str(e)}",
            "success": False
        }), 500


@app.route('/upload', methods=['POST'])
def upload_pdf():
    """رفع ملف PDF جديد وبناء قاعدة البيانات"""
    global qa_system
    
    if 'file' not in request.files:
        return jsonify({"error": "لم يتم رفع ملف"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "اسم الملف فارغ"}), 400
    
    if not file.filename.endswith('.pdf'):
        return jsonify({"error": "يجب أن يكون الملف PDF"}), 400
    
    try:
        # حفظ الملف مؤقتاً
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            file.save(tmp_file.name)
            pdf_path = tmp_file.name
        
        print(f"📄 تحميل ملف: {file.filename}")
        
        # قراءة PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        print(f"✅ تم قراءة {len(documents)} صفحة")
        
        # تقسيم النص
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        print(f"✅ تم تقسيم النص إلى {len(chunks)} قطعة")
        
        # إنشاء Embeddings وقاعدة البيانات
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # حذف قاعدة البيانات القديمة إن وجدت
        if os.path.exists(CHROMA_DIR):
            import shutil
            shutil.rmtree(CHROMA_DIR)
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
        vectorstore.persist()
        
        print("✅ تم بناء قاعدة البيانات")
        
        # تحديث النظام
        initialize_db()
        
        # حذف الملف المؤقت
        os.unlink(pdf_path)
        
        return jsonify({
            "success": True,
            "message": f"تم معالجة الملف بنجاح! ({len(chunks)} قطعة نصية)",
            "pages": len(documents),
            "chunks": len(chunks)
        })
    
    except Exception as e:
        return jsonify({
            "error": f"حدث خطأ أثناء معالجة الملف: {str(e)}",
            "success": False
        }), 500


if __name__ == '__main__':
    # محاولة تحميل قاعدة البيانات عند بدء التشغيل
    initialize_db()
    
    # تشغيل السيرفر
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)