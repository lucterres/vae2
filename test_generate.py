"""
Script de teste rápido - Gera apenas 5 máscaras para validar o workflow
"""

from generate_masks import *

if __name__ == "__main__":
    print("="*60)
    print("Teste Rápido - Gerador de Máscaras")
    print("="*60)
    
    # Configurações de teste
    NUM_SAMPLES = 5
    OUTPUT_DIR = './test_masks'
    
    # Cria diretório
    output_dir = create_output_directory(OUTPUT_DIR)
    
    # Carrega modelo (sem checkpoint para teste)
    print("\nCarregando modelo (modo teste - sem checkpoint)...")
    decoder = load_model(checkpoint_path=None)
    
    # Gera apenas 5 máscaras
    print("\nGerando 5 máscaras de teste...")
    generate_masks(decoder, num_samples=NUM_SAMPLES, output_dir=output_dir)
    
    # Cria visualização
    create_visualization(output_dir, num_display=5)
    
    print("\n" + "="*60)
    print("Teste concluído!")
    print(f"Verifique as máscaras em: {output_dir}")
    print("="*60)
    print("\nNOTA: Máscaras geradas com modelo NÃO treinado")
    print("Para resultados reais, treine o modelo no notebook primeiro")
