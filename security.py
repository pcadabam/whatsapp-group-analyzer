import os
import zipfile
import tempfile
from pathlib import Path
import mimetypes
import hashlib

# Optional python-magic import with fallback
try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False

class SecurityError(Exception):
    """Custom exception for security violations"""
    pass

class FileSecurityValidator:
    def __init__(self):
        # Maximum file sizes (in bytes)
        self.MAX_ZIP_SIZE = 50 * 1024 * 1024  # 50MB
        self.MAX_TXT_SIZE = 20 * 1024 * 1024  # 20MB
        self.MAX_EXTRACTED_SIZE = 100 * 1024 * 1024  # 100MB total extracted
        
        # Allowed file types
        self.ALLOWED_ZIP_TYPES = ['application/zip', 'application/x-zip-compressed']
        self.ALLOWED_TXT_TYPES = ['text/plain', 'text/txt']
        
        # Dangerous file extensions/patterns
        self.BLOCKED_EXTENSIONS = {
            '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.vbe', 
            '.js', '.jse', '.jar', '.msi', '.dll', '.sys', '.bin', '.sh', 
            '.ps1', '.py', '.php', '.jsp', '.asp', '.aspx', '.html', '.htm'
        }
        
        # Suspicious filename patterns
        self.SUSPICIOUS_PATTERNS = [
            '../', '..\\', '/', '\\', ':', '*', '?', '"', '<', '>', '|',
            'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 
            'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 
            'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        ]
        
        # WhatsApp chat patterns for validation
        self.WHATSAPP_PATTERNS = [
            r'\[\d{2}/\d{2}/\d{2}',  # [DD/MM/YY
            r'\d{1,2}:\d{2}:\d{2}\s[AP]M',  # HH:MM:SS AM/PM
            r':\s',  # Message separator
            'WhatsApp',  # App name sometimes in exports
        ]
    
    def validate_file_size(self, uploaded_file):
        """Validate file size limits"""
        if uploaded_file.size > self.MAX_ZIP_SIZE:
            raise SecurityError(f"File too large. Maximum size allowed: {self.MAX_ZIP_SIZE // 1024 // 1024}MB")
    
    def validate_file_type(self, uploaded_file):
        """Validate file type using multiple methods"""
        filename = uploaded_file.name.lower()
        
        # Check file extension
        if filename.endswith('.zip'):
            expected_types = self.ALLOWED_ZIP_TYPES
        elif filename.endswith('.txt'):
            expected_types = self.ALLOWED_TXT_TYPES
        else:
            raise SecurityError("Only .zip and .txt files are allowed")
        
        # Validate MIME type from Streamlit
        if hasattr(uploaded_file, 'type') and uploaded_file.type:
            if uploaded_file.type not in expected_types:
                raise SecurityError(f"Invalid file type: {uploaded_file.type}")
    
    def validate_filename(self, filename):
        """Check for suspicious filename patterns"""
        filename_lower = filename.lower()
        
        # Check for blocked extensions
        file_ext = Path(filename_lower).suffix
        if file_ext in self.BLOCKED_EXTENSIONS:
            raise SecurityError(f"Blocked file extension: {file_ext}")
        
        # Check for suspicious patterns
        for pattern in self.SUSPICIOUS_PATTERNS:
            if pattern.lower() in filename_lower:
                raise SecurityError(f"Suspicious filename pattern detected: {pattern}")
        
        # Check filename length
        if len(filename) > 255:
            raise SecurityError("Filename too long")
    
    def validate_zip_contents(self, zip_path):
        """Safely validate ZIP file contents"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                if len(file_list) == 0:
                    raise SecurityError("Empty ZIP file")
                
                if len(file_list) > 50:  # Reasonable limit for WhatsApp exports
                    raise SecurityError("ZIP contains too many files")
                
                total_size = 0
                chat_files_found = 0
                
                for file_info in zip_ref.infolist():
                    filename = file_info.filename
                    file_size = file_info.file_size
                    
                    # Validate each filename
                    self.validate_filename(filename)
                    
                    # Check for directory traversal
                    if os.path.isabs(filename) or ".." in filename:
                        raise SecurityError(f"Unsafe file path: {filename}")
                    
                    # Check individual file size
                    if file_size > self.MAX_TXT_SIZE:
                        raise SecurityError(f"File too large in ZIP: {filename}")
                    
                    # Track total extracted size
                    total_size += file_size
                    if total_size > self.MAX_EXTRACTED_SIZE:
                        raise SecurityError("ZIP contents too large when extracted")
                    
                    # Count potential chat files
                    if filename.lower().endswith(('.txt', '_chat.txt')):
                        chat_files_found += 1
                
                if chat_files_found == 0:
                    raise SecurityError("No text files found in ZIP (expected WhatsApp chat export)")
                
                return True
                
        except zipfile.BadZipFile:
            raise SecurityError("Corrupted or invalid ZIP file")
        except Exception as e:
            if isinstance(e, SecurityError):
                raise
            raise SecurityError(f"Error validating ZIP file: {str(e)}")
    
    def validate_text_content(self, content, max_sample_size=10000):
        """Validate text content looks like WhatsApp chat"""
        if not content or len(content.strip()) == 0:
            raise SecurityError("Empty file content")
        
        # Sample first part of content for validation
        sample = content[:max_sample_size]
        
        # Check for binary content (should be text)
        try:
            sample.encode('utf-8')
        except UnicodeEncodeError:
            raise SecurityError("File contains non-text content")
        
        # Look for WhatsApp chat patterns
        import re
        whatsapp_indicators = 0
        for pattern in self.WHATSAPP_PATTERNS:
            if re.search(pattern, sample, re.IGNORECASE):
                whatsapp_indicators += 1
        
        # Should have at least some WhatsApp indicators
        if whatsapp_indicators < 2:
            # Allow it but warn - might be a different format
            pass
        
        # Check for suspicious script content
        suspicious_keywords = [
            '<script>', 'javascript:', 'eval(', 'exec(', 'import os', 
            'import subprocess', 'rm -rf', 'del /f', '__import__',
            'system(', 'shell_exec', 'passthru', 'exec(', 'popen('
        ]
        
        content_lower = content.lower()
        for keyword in suspicious_keywords:
            if keyword in content_lower:
                raise SecurityError(f"Suspicious content detected: {keyword}")
    
    def scan_file_for_malware_signatures(self, file_path):
        """Basic malware signature detection"""
        # Simple signature-based detection
        malware_signatures = [
            b'MZ\x90\x00',  # PE executable header
            b'\x7fELF',     # ELF executable header
            b'\xca\xfe\xba\xbe',  # Java class file
            b'PK\x03\x04',  # ZIP header (we allow this)
        ]
        
        dangerous_signatures = [
            b'MZ\x90\x00',  # PE executable
            b'\x7fELF',     # ELF executable  
            b'\xca\xfe\xba\xbe',  # Java class
        ]
        
        try:
            with open(file_path, 'rb') as f:
                header = f.read(1024)  # Read first 1KB
                
                for sig in dangerous_signatures:
                    if sig in header:
                        raise SecurityError("Executable file detected")
                        
        except Exception as e:
            if isinstance(e, SecurityError):
                raise
            # If we can't read the file, that's suspicious too
            raise SecurityError("Cannot validate file integrity")
    
    def create_secure_temp_file(self, content, suffix='.txt'):
        """Create a secure temporary file"""
        # Create in a secure temporary directory
        temp_dir = tempfile.mkdtemp(prefix='whatsapp_secure_')
        
        # Generate secure filename
        file_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        temp_path = os.path.join(temp_dir, f'chat_{file_hash}{suffix}')
        
        # Write with restricted permissions
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Set restrictive permissions (owner read/write only)
        os.chmod(temp_path, 0o600)
        
        return temp_path
    
    def secure_cleanup(self, file_paths):
        """Securely delete temporary files"""
        for file_path in file_paths if isinstance(file_paths, list) else [file_paths]:
            try:
                if os.path.exists(file_path):
                    # Overwrite file content before deletion (basic secure delete)
                    file_size = os.path.getsize(file_path)
                    if file_size > 0:
                        with open(file_path, 'r+b') as f:
                            f.write(b'\x00' * file_size)
                            f.flush()
                            os.fsync(f.fileno())
                    
                    os.unlink(file_path)
                    
                    # Also try to remove parent temp directory if empty
                    parent_dir = os.path.dirname(file_path)
                    if parent_dir.startswith(tempfile.gettempdir()):
                        try:
                            os.rmdir(parent_dir)
                        except OSError:
                            pass  # Directory not empty
            except Exception:
                pass  # Best effort cleanup
    
    def validate_uploaded_file(self, uploaded_file):
        """Complete validation pipeline for uploaded files"""
        try:
            # Step 1: Basic file validation
            self.validate_file_size(uploaded_file)
            self.validate_file_type(uploaded_file)
            self.validate_filename(uploaded_file.name)
            
            # Step 2: If it's a ZIP, validate contents
            if uploaded_file.name.lower().endswith('.zip'):
                # Save to temporary file for ZIP validation
                with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                    tmp_zip.write(uploaded_file.getvalue())
                    tmp_zip_path = tmp_zip.name
                
                try:
                    # Scan for malware signatures
                    self.scan_file_for_malware_signatures(tmp_zip_path)
                    
                    # Validate ZIP contents
                    self.validate_zip_contents(tmp_zip_path)
                    
                finally:
                    # Clean up temp file
                    self.secure_cleanup(tmp_zip_path)
            
            # Step 3: If it's a text file, validate content
            elif uploaded_file.name.lower().endswith('.txt'):
                content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
                self.validate_text_content(content)
            
            return True
            
        except SecurityError:
            raise
        except Exception as e:
            raise SecurityError(f"File validation failed: {str(e)}")

# Create global validator instance
file_validator = FileSecurityValidator()