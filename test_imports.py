"""Test that all modules can be imported successfully."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test all module imports."""
    errors = []

    # Test core modules
    modules_to_test = [
        ('src.config', 'Settings'),
        ('src.database', 'Base'),
        ('src.models', 'Persona'),
        ('src.models', 'ContentChunk'),
        ('src.crawlers.base', 'BaseCrawler'),
        ('src.crawlers.youtube', 'YouTubeCrawler'),
        ('src.crawlers.github', 'GitHubCrawler'),
        ('src.crawlers.registry', 'get_crawler'),
        ('src.vector_store.chunking', 'chunk_text'),
        ('src.vector_store.embeddings', 'generate_embedding'),
        ('src.vector_store.store', 'VectorStore'),
        ('src.agents.persona_agent', 'persona_agent'),
        ('src.api.app', 'app'),
    ]

    print("Testing module imports...")
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✓ {module_name}.{class_name}")
        except ImportError as e:
            errors.append(f"✗ {module_name}.{class_name}: {e}")
            print(f"✗ {module_name}.{class_name}: Import Error")
        except AttributeError as e:
            errors.append(f"✗ {module_name}.{class_name}: {e}")
            print(f"✗ {module_name}.{class_name}: Attribute Error")
        except Exception as e:
            errors.append(f"✗ {module_name}.{class_name}: {type(e).__name__}: {e}")
            print(f"✗ {module_name}.{class_name}: {type(e).__name__}")

    print(f"\n{'='*60}")
    if errors:
        print(f"FAILED: {len(errors)} import errors")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print(f"SUCCESS: All {len(modules_to_test)} modules imported successfully!")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
