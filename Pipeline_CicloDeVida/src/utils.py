# Funciones comunes (logging, helpers)
import logging
import sys
from pathlib import Path
from datetime import datetime

class DualFileWriter:
    """Escribe salida tanto en archivo como en consola simultáneamente.
    
    Útil para capturar stdout/stderr en un archivo de log mientras se
    mantiene la visualización en consola, con manejo robusto de encoding.
    """
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
    
    def write(self, message):
        try:
            # Escribir en archivo con UTF-8 completo
            self.file.write(message)
            self.file.flush()
        except Exception as e:
            pass
        
        try:
            # Escribir en consola intentando con UTF-8
            self.stdout.write(message)
            self.stdout.flush()
        except UnicodeEncodeError:
            # Si hay error de codificación, reemplazar caracteres especiales
            try:
                clean_message = message.encode('ascii', 'replace').decode('ascii')
                self.stdout.write(clean_message)
                self.stdout.flush()
            except:
                pass
    
    def flush(self):
        try:
            self.file.flush()
        except:
            pass
        try:
            self.stdout.flush()
        except:
            pass
    
    def close(self):
        try:
            self.file.close()
        except:
            pass

def setup_logger(name, log_level=logging.INFO):
    """Configura un logger con manejo de formato y streaming.
    
    Args:
        name: Nombre del logger (usualmente __name__).
        log_level: Nivel mínimo de logging (por defecto INFO).
    
    Returns:
        Logger configurado listo para usar.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def setup_results_output(results_dir):
    """Redirige stdout y stderr a un archivo de log con timestamp.
    
    Permite capturar toda la salida de prints en un archivo txt mientras
    se mantiene visible en consola, útil para documentar ejecuciones.
    
    Args:
        results_dir: Directorio donde guardar el archivo de log.
    
    Returns:
        Tupla (path_archivo_log, dual_writer_instance) para posible limpieza posterior.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear nombre con timestamp (formato: YYYYMMDD_HHMMSS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = results_dir / f"execution_log_{timestamp}.txt"
    
    # Redirigir stdout y stderr
    dual_writer = DualFileWriter(log_file)
    sys.stdout = dual_writer
    sys.stderr = dual_writer
    
    return log_file, dual_writer
